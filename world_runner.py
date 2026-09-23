# -*- coding: utf-8 -*-
"""
Save decoding, as its own program: `python -m world_runner <job_id>`.

Same reasoning as ultra_runner - the parsed world and the job's progress both
have to be visible to every gunicorn worker, not just the one that took the
upload, so the child writes them to the job store itself.
"""
import hashlib
import os
import sys
import threading
import time
import traceback

import job_store


def main(job_id):
    payload = job_store.read_json(job_id, "payload") or {}
    path = payload.get("path") or ""
    original = payload.get("original") or "world.sav"
    cleanup = bool(payload.get("cleanup"))

    status = {"state": "running", "percent": 0.0, "stage": "Starting",
              "error": None, "started": payload.get("started") or time.time(),
              "pid": os.getpid()}
    last = [0.0]

    def flush(force=False):
        now = time.monotonic()
        if not force and now - last[0] < 0.2:
            return
        last[0] = now
        job_store.write_json(job_id, "status", status)

    flush(force=True)

    # Decoding a 15MB save runs ~45s inside one stage, so without this the
    # status file would look stale and be mistaken for a dead child.
    done = threading.Event()

    def heartbeat():
        while not done.wait(5.0):
            status["alive"] = time.time()
            try:
                flush(force=True)
            except Exception:                             # noqa: BLE001
                pass

    threading.Thread(target=heartbeat, daemon=True).start()

    try:
        from save_parser import parse_save

        def progress(stage, frac):
            status["stage"] = stage
            status["percent"] = round(float(frac) * 100, 1)
            flush()

        world = parse_save(path, progress=progress)

        world_id = hashlib.sha1(
            f"{original}{len(world['recipes'])}{time.time()}".encode()
        ).hexdigest()[:12]
        job_store.put_world(world_id, world)

        job_store.write_json(job_id, "result", {
            "world_id":   world_id,
            "name":       os.path.splitext(original)[0] or "World",
            "parser":     world["parser"],
            "schematics": len(world["schematics"]),
            "recipes":    world["recipes"],
            "machines":   world["machines"],
            "resources":  world["resources"],
            "unknown":    world["unknown_schematics"],
        })
        status["state"] = "done"
        status["percent"] = 100.0
        status["stage"] = "Done"
    except Exception as exc:                              # noqa: BLE001
        status["state"] = "error"
        status["error"] = f"Could not read save: {exc}"
        try:
            job_store.write_json(job_id, "traceback",
                                 {"tb": traceback.format_exc()})
        except Exception:                                 # noqa: BLE001
            pass
    finally:
        done.set()
        status["ended"] = time.time()
        status["alive"] = time.time()
        flush(force=True)
        if cleanup and path:
            try:
                os.unlink(path)
            except OSError:
                pass


def _run_and_quit(job_id):
    """
    Do the work, then leave immediately.

    A normal return waits on every non-daemon thread, and the solver leaves a
    pile of them behind - a search that had been cancelled carried on burning a
    core for over two hours after it had already written its result. Everything
    this process owns is flushed to the job store before this point, and those
    writes are atomic, so there is nothing left to tidy.
    """
    try:
        main(job_id)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m world_runner <job_id>", file=sys.stderr)
        raise SystemExit(2)
    _run_and_quit(sys.argv[1])
