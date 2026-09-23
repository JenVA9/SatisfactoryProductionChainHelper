# -*- coding: utf-8 -*-
"""
The ULTRA search, as its own program: `python -m ultra_runner <job_id>`.

Launched with subprocess rather than multiprocessing on purpose. Under gunicorn
`__main__` is the gunicorn console script, and multiprocessing's spawn start
method re-imports `__main__` in the child - which would try to boot a second
gunicorn. A plain `-m` subprocess has no such fixup, and it works the same on
Windows and Linux.

The child writes its own progress into the job directory, so no web worker owns
the job and any of them can report on it.
"""
import json
import os
import sys
import threading
import time
import traceback

import job_store


STATUS_EVERY = 0.25          # seconds; the page polls once a second


def main(job_id):
    started = time.time()
    payload = job_store.read_json(job_id, "payload") or {}
    cancelled = job_store.stop_watcher(job_id)

    status = {"state": "running", "checked": 0, "total": 0, "stage": "Starting",
              "best": None, "error": None,
              "started": payload.get("started") or started,
              "pid": os.getpid()}
    last_write = [0.0]

    def flush(force=False):
        now = time.monotonic()
        if not force and now - last_write[0] < STATUS_EVERY:
            return
        last_write[0] = now
        job_store.write_json(job_id, "status", status)

    flush(force=True)

    # A single refined candidate can solve for minutes without emitting
    # anything, so the file's own mtime is the liveness signal - otherwise the
    # server cannot tell a long solve from a child that died.
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
        import calculator

        world = job_store.get_world(payload.get("world_id") or "")

        def progress(checked, total, best_stats, stage=None):
            status["checked"] = checked
            status["total"] = total
            if stage is not None:
                status["stage"] = stage
            if best_stats:
                status["best"] = {k: best_stats.get(k) for k in
                                  ("power_mw", "machines", "steps", "byproducts",
                                   "byproduct_total", "byproduct_fluid_total",
                                   "nice_machines", "machine_groups", "outputs")}
            flush()

        def on_best(result):
            # Snapshot each leader so Keep can answer from any worker at once.
            # Marked here because ultra_search only stamps its summary on the
            # winner at the end, and a kept plan must not look like a plain
            # calculation.
            live = calculator._client_payload(result)
            meta = dict(live["meta"])
            meta["ultra"] = True
            meta["ultra_partial"] = True
            live["meta"] = meta
            job_store.write_json(job_id, "live", live)

        result = calculator.ultra_search(
            payload.get("targets") or [],
            mode=payload.get("mode", "least_machines"),
            world=world,
            blocked_recipes=payload.get("blocked_recipes") or (),
            blocked_machines=payload.get("blocked_machines") or (),
            resource_limits=payload.get("resource_limits") or None,
            least_resources=payload.get("least_resources") or "off",
            name=payload.get("name"),
            priority=payload.get("priority") or None,
            ban_depth=payload.get("ultra_depth"),
            workers=payload.get("workers"),
            progress=progress, on_best=on_best, cancelled=cancelled)

        if result is not None:
            job_store.write_json(job_id, "result",
                                 calculator._client_payload(result))
            status["state"] = "cancelled" if cancelled() else "done"
        elif cancelled():
            status["state"] = "cancelled"
        else:
            status["state"] = "error"
            status["error"] = "No workable plan found."
    except Exception as exc:                              # noqa: BLE001
        status["state"] = "error"
        status["error"] = f"{exc}"
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
        print("usage: python -m ultra_runner <job_id>", file=sys.stderr)
        raise SystemExit(2)
    _run_and_quit(sys.argv[1])
