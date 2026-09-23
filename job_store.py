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
WORLDS_ROOT = os.path.join(JOBS_ROOT, "worlds")

# How long a finished job's files stick around for.
JOB_TTL = 15 * 60
WORLD_TTL = 12 * 60 * 60


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

def request_stop(job_id: str) -> None:
    try:
        open(os.path.join(_ensure(job_dir(job_id)), "cancel"), "w").close()
    except OSError:
        pass


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


# --- worlds -----------------------------------------------------------------
# A parsed save is read by whichever worker happens to serve the next calculate,
# so it cannot live in the importing worker's memory either.

_world_memo = {}


def put_world(world_id: str, world: dict) -> None:
    _ensure(WORLDS_ROOT)
    fd, tmp = tempfile.mkstemp(dir=WORLDS_ROOT, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(world, f)
        os.replace(tmp, os.path.join(WORLDS_ROOT, world_id + ".json"))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def get_world(world_id: str):
    if not world_id or not world_id.isalnum():
        return None
    hit = _world_memo.get(world_id)
    if hit is not None:
        return hit
    path = os.path.join(WORLDS_ROOT, world_id + ".json")
    try:
        with open(path, encoding="utf-8") as f:
            world = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if len(_world_memo) > 8:                 # per worker; the file is the truth
        _world_memo.clear()
    _world_memo[world_id] = world
    return world


# --- housekeeping -----------------------------------------------------------

def prune() -> None:
    """Drop finished jobs and stale worlds. Cheap enough to call on each start."""
    now = time.time()
    try:
        entries = os.listdir(JOBS_ROOT)
    except OSError:
        return
    for name in entries:
        path = os.path.join(JOBS_ROOT, name)
        if name == "worlds" or not os.path.isdir(path):
            continue
        try:
            if now - os.path.getmtime(path) > JOB_TTL:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass
    try:
        for name in os.listdir(WORLDS_ROOT):
            path = os.path.join(WORLDS_ROOT, name)
            try:
                if now - os.path.getmtime(path) > WORLD_TTL:
                    os.unlink(path)
            except OSError:
                pass
    except OSError:
        pass
