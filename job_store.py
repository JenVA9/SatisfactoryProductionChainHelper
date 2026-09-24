# -*- coding: utf-8 -*-
"""
Job state that every worker can see.

Under gunicorn each worker is a separate process with its own memory, so a job
kept in a module-level dict only exists for the worker that created it. With
nine workers a status poll found its job about one time in nine and otherwise
answered "Unknown or expired search."

So the state lives on disk instead, one directory per job, and the worker that
started the job owns nothing: the *child process* doing the work writes its own
progress, and any worker can read it or ask it to stop.

Writes are atomic (temp file in the same directory, then os.replace), so a
reader never sees a half-written file.
"""
import json
import os
import shutil
import tempfile
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS_ROOT = os.environ.get("JOBS_DIR") or os.path.join(HERE, ".jobs")

# How long a finished job's files stick around for.
JOB_TTL = 15 * 60


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def job_dir(job_id: str) -> str:
    # uuid hex only, so a job id can never climb out of the jobs directory.
    if not job_id or not job_id.isalnum():
        raise ValueError("bad job id")
    return os.path.join(JOBS_ROOT, job_id)


def create(job_id: str) -> str:
    return _ensure(job_dir(job_id))


def exists(job_id: str) -> bool:
    try:
        return os.path.isdir(job_dir(job_id))
    except ValueError:
        return False


def write_json(job_id: str, name: str, obj) -> None:
    """Replace <job>/<name>.json atomically."""
    d = _ensure(job_dir(job_id))
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(tmp, os.path.join(d, name + ".json"))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(job_id: str, name: str):
    """Return the object, or None if it is not there (or not readable yet)."""
    try:
        path = os.path.join(job_dir(job_id), name + ".json")
    except ValueError:
        return None
    for _ in range(3):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError):
            # Only possible on a platform where replace is not atomic; a retry
            # costs nothing and beats handing the browser a 500.
            time.sleep(0.02)
    return None


# --- stopping ---------------------------------------------------------------
# A flag file rather than a signal: the worker asked to cancel is usually not
# the one that started the job, so it has nothing to signal.

STOP_GRACE = 20          # seconds a child gets to stop before it is killed


def request_stop(job_id: str) -> None:
    try:
        with open(os.path.join(_ensure(job_dir(job_id)), "cancel"), "w") as f:
            f.write(str(time.time()))
    except OSError:
        pass


def _pid_is_this_job(pid: int, job_id: str) -> bool:
    """
    Is `pid` still the runner for this job?

    Checked against the process's own command line rather than trusting the
    number: pids get reused, and killing an unrelated process would be far
    worse than leaving a stray one running.
    """
    try:
        if os.name == "nt":
            # PowerShell, not wmic: wmic is deprecated and absent from recent
            # Windows builds, and its FileNotFoundError was being swallowed -
            # so this always said "not my process" and nothing was ever killed.
            import subprocess
            out = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}')"
                 f".CommandLine"],
                capture_output=True, text=True, timeout=20).stdout
            return job_id in out
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return job_id.encode() in f.read()
    except Exception:                                     # noqa: BLE001
        return False


def enforce_stop(job_id: str, grace: float = STOP_GRACE) -> bool:
    """
    Kill a child that was asked to stop and did not.

    The worker that asks for the stop is usually not the one that started the
    job, so nobody owns the process - and a search whose solver threads are
    still running will not exit on its own. Any worker that notices an overdue
    job can end it.
    """
    try:
        path = os.path.join(job_dir(job_id), "cancel")
        with open(path, encoding="utf-8") as f:
            asked = float((f.read() or "0").strip() or 0)
    except (OSError, ValueError):
        return False
    if not asked or time.time() - asked < grace:
        return False

    st = read_json(job_id, "status") or {}
    pid = st.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    if not _pid_is_this_job(pid, job_id):
        # Gone already (or the pid was recycled). Mark it so later polls stop
        # paying for a process lookup on every request.
        st["reaped"] = True
        try:
            write_json(job_id, "status", st)
        except OSError:
            pass
        return False
    try:
        if os.name == "nt":
            import subprocess
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True, timeout=10)
        else:
            import signal
            os.kill(pid, signal.SIGKILL)
    except Exception:                                     # noqa: BLE001
        return False

    st["state"] = "cancelled"
    st["reaped"] = True
    st["ended"] = time.time()
    st["alive"] = time.time()
    try:
        write_json(job_id, "status", st)
    except OSError:
        pass
    return True


def stop_requested(job_id: str) -> bool:
    try:
        return os.path.exists(os.path.join(job_dir(job_id), "cancel"))
    except ValueError:
        return False


def stop_watcher(job_id: str, every: float = 0.25):
    """A cheap `cancelled()` for the solver - stats the flag at most 4x/sec."""
    state = {"at": 0.0, "hit": False}

    def cancelled():
        if state["hit"]:
            return True
        now = time.monotonic()
        if now - state["at"] >= every:
            state["at"] = now
            state["hit"] = stop_requested(job_id)
        return state["hit"]

    return cancelled


# --- housekeeping -----------------------------------------------------------

LIVE_GRACE = 120        # a job whose heartbeat is newer than this is alive

# Set when a sweep could not remove something, so the condition is observable
# instead of silently swallowed. Read it from /api/health.
prune_failures = 0
prune_last_error = None


def _job_is_live(job_id: str) -> bool:
    """
    Would deleting this job's files orphan a running child?

    Anything still marked running with a recent heartbeat is left alone. Its
    pid lives in status.json, and that file is the only way anything can ever
    stop it.
    """
    st = read_json(job_id, "status")
    if not st:
        return False
    if st.get("state") not in ("running",):
        return False
    beat = float(st.get("alive") or st.get("started") or 0)
    return (time.time() - beat) < LIVE_GRACE


def _remove_tree(path: str) -> bool:
    """
    Delete a directory, working around a holder that lets go a moment later.

    OneDrive (and any indexer) can keep a handle on a folder it has just
    written, which makes rmtree fail outright even when the folder is empty.
    Clearing the contents first and retrying the directory itself gets past it.
    """
    for attempt in range(4):
        try:
            for name in os.listdir(path):
                try:
                    os.unlink(os.path.join(path, name))
                except IsADirectoryError:
                    shutil.rmtree(os.path.join(path, name), ignore_errors=True)
                except OSError:
                    pass
            os.rmdir(path)
            return True
        except FileNotFoundError:
            return True
        except OSError as exc:
            if attempt == 3:
                if os.name == "nt" and _nt_force_remove(path):
                    return True
                global prune_failures, prune_last_error
                prune_failures += 1
                prune_last_error = f"{type(exc).__name__}: {exc}"
                return False
            time.sleep(0.15 * (attempt + 1))
    return False


def _nt_force_remove(path: str) -> bool:
    """
    Last resort on Windows only.

    OneDrive keeps a handle on a folder it has just synced, and `os.rmdir`
    never gets past it - not on a retry, not minutes later. PowerShell's
    Remove-Item clears the same folders immediately. Linux, where this actually
    runs in production, never needs this.
    """
    try:
        import subprocess
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             f"Remove-Item -LiteralPath '{path}' -Recurse -Force "
             f"-ErrorAction SilentlyContinue"],
            capture_output=True, timeout=20)
        return not os.path.exists(path)
    except Exception:                                     # noqa: BLE001
        return False


def prune() -> None:
    """Drop finished jobs. Cheap enough to call on each start."""
    now = time.time()
    try:
        entries = os.listdir(JOBS_ROOT)
    except OSError:
        return
    for name in entries:
        path = os.path.join(JOBS_ROOT, name)
        if not os.path.isdir(path):
            continue
        try:
            if now - os.path.getmtime(path) <= JOB_TTL:
                continue
        except OSError:
            continue
        if not name.isalnum() or _job_is_live(name):
            continue                     # its child may still need that pid
        _remove_tree(path)
