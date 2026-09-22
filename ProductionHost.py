# -*- coding: utf-8 -*-
"""
ProductionHost.py
Flask server for the Satisfactory Production Chain Helper.
Serves main.html, the share-link solver, and the production calculator.
"""

import os
import queue
import sys
import json
import time
import hashlib
import tempfile
import threading

from flask import Flask, send_from_directory, request, jsonify

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Sanity check - make sure ShareCodeResolver is available before we start
# ---------------------------------------------------------------------------

if not os.path.exists(os.path.join(HERE, "ShareCodeResolver.py")):
    print("[ERROR] ShareCodeResolver.py not found in the same directory.")
    print("        Make sure both files are in the same folder before running.")
    sys.exit(1)

try:
    from ShareCodeResolver import get_production_chain
except ImportError as e:
    print(f"[ERROR] Could not import ShareCodeResolver: {e}")
    print("        Run: pip install requests")
    sys.exit(1)

# Calculator is optional - the share-link half of the app still works without
# scipy installed, so we degrade rather than refuse to boot.
try:
    import calculator
    from docs_parser import load_docs
    CALC_ERROR = None
except Exception as e:                                    # noqa: BLE001
    calculator = None
    CALC_ERROR = str(e)
    print(f"[WARN] Calculator unavailable: {e}")

try:
    from save_parser import parse_save
    SAVE_ERROR = None
except Exception as e:                                    # noqa: BLE001
    parse_save = None
    SAVE_ERROR = str(e)
    print(f"[WARN] Save import unavailable: {e}")

app = Flask(__name__, static_folder=HERE)

# ---------------------------------------------------------------------------
# In-memory world store + on-disk result cache
# ---------------------------------------------------------------------------

_worlds = {}                       # world_id -> parsed world dict
_worlds_lock = threading.Lock()

CACHE_PATH = os.path.join(HERE, ".calc_cache.json")
_cache_lock = threading.Lock()


def _load_cache() -> dict:
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:                                     # noqa: BLE001
        return {}


_calc_cache = _load_cache()


def _cache_key(payload: dict) -> str:
    """Stable key for a calculation request (world included by id)."""
    blob = json.dumps({
        "targets": payload.get("targets"),
        "mode":    payload.get("mode"),
        "depth":   payload.get("depth"),
        "world":   payload.get("world_id"),
        "br":      sorted(payload.get("blocked_recipes") or []),
        "bm":      sorted(payload.get("blocked_machines") or []),
        "rl":      sorted((payload.get("resource_limits") or {}).items()),
        "lr":      payload.get("least_resources") or "off",
        "pr":      payload.get("priority") or [],
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _cache_store(key: str, value: dict):
    with _cache_lock:
        _calc_cache[key] = value
        # Keep the file small; this is a convenience cache, not a database.
        if len(_calc_cache) > 400:
            for k in list(_calc_cache)[:100]:
                _calc_cache.pop(k, None)
        try:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(_calc_cache, f)
        except Exception:                                 # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# ULTRA jobs - long searches run in the background so the request can return
# and the browser can poll for progress.
# ---------------------------------------------------------------------------

_ultra_jobs = {}
_ultra_lock = threading.Lock()


def _ultra_prune():
    """Drop finished jobs nobody has collected for a while."""
    now = time.time()
    for jid, job in list(_ultra_jobs.items()):
        if job["state"] in ("done", "error", "cancelled") and now - job["ended"] > 900:
            _ultra_jobs.pop(jid, None)


def _run_ultra(job_id, body, world):
    """
    Drive a search running in its own process.

    This thread only moves messages off a queue, so it is idle almost all the
    time and the web server keeps its interpreter to itself. Everything heavy -
    model building, HiGHS, graph layout - happens in the child.
    """
    job = _ultra_jobs[job_id]

    # Share the machine between concurrent searches rather than letting each one
    # size its own pool against the full core count.
    with _ultra_lock:
        running = max(1, sum(1 for j in _ultra_jobs.values()
                             if j["state"] == "running"))
    workers = max(2, min(8, calculator.ULTRA_MAX_CONCURRENT_SOLVES // running))

    payload = {
        "targets":          body.get("targets") or [],
        "mode":             body.get("mode", "least_machines"),
        "world":            world,
        "blocked_recipes":  body.get("blocked_recipes") or (),
        "blocked_machines": body.get("blocked_machines") or (),
        "resource_limits":  body.get("resource_limits") or None,
        "least_resources":  body.get("least_resources") or "off",
        "name":             body.get("name"),
        "priority":         body.get("priority") or None,
        "ban_depth":        body.get("ultra_depth"),   # NOT "depth"
        "workers":          workers,
    }

    proc = None
    try:
        proc, q, cancel_ev = calculator.start_ultra_process(payload)
        job["_proc"] = proc

        while True:
            if job["cancel"] and not cancel_ev.is_set():
                cancel_ev.set()
                job["_cancel_at"] = time.time()

            try:
                msg = q.get(timeout=0.4)
            except queue.Empty:
                if not proc.is_alive():
                    break
                # Cooperative cancel first, then insist. An in-flight solve
                # cannot be interrupted from outside, and during refinement one
                # can hold on for minutes.
                if job["cancel"] and time.time() - job.get("_cancel_at", 0) > 3:
                    proc.terminate()
                    break
                continue

            kind = msg[0]
            if kind == "progress":
                _, checked, total, best_stats, stage = msg
                job["checked"], job["total"] = checked, total
                if stage is not None:
                    job["stage"] = stage
                if best_stats:
                    job["best"] = {k: best_stats.get(k) for k in
                                   ("power_mw", "machines", "steps", "byproducts",
                                    "byproduct_total", "byproduct_fluid_total",
                                    "nice_machines", "machine_groups", "outputs")}
            elif kind == "best":
                # Snapshot every leader so Keep can answer instantly. Mark it,
                # or a kept plan comes back looking like a plain calculation.
                live = msg[1]
                meta = dict(live["meta"])
                meta["ultra"] = True
                meta["ultra_partial"] = True
                live["meta"] = meta
                job["live_result"] = live
            elif kind == "done":
                job["result"] = msg[1]
                break
            elif kind == "error":
                job["error"] = msg[1]
                break

        if job["error"]:
            job["state"] = "error"
        elif job["cancel"]:
            job["state"] = "cancelled"
        elif job["result"] is None:
            job["state"] = "error"
            job["error"] = "No workable plan found."
        else:
            job["state"] = "done"
    except Exception as e:                                # noqa: BLE001
        job["state"] = "error"
        job["error"] = str(e)
    finally:
        job["ended"] = time.time()
        job.pop("_proc", None)
        if proc is not None and proc.is_alive():
            proc.terminate()


@app.route("/api/ultra", methods=["POST"])
def ultra_start():
    if calculator is None:
        return jsonify({"error": f"Calculator unavailable: {CALC_ERROR}"}), 503
    body = request.get_json(silent=True) or {}
    if not (body.get("targets") or []):
        return jsonify({"error": "Add at least one target item."}), 400

    world = None
    if body.get("world_id"):
        with _worlds_lock:
            world = _worlds.get(body["world_id"])
        if world is None:
            return jsonify({"error": "World not loaded. Re-import your save."}), 400

    job_id = hashlib.sha1(f"{time.time()}{id(body)}".encode()).hexdigest()[:12]
    with _ultra_lock:
        _ultra_prune()
        if sum(1 for j in _ultra_jobs.values() if j["state"] == "running") >= 3:
            return jsonify({"error": "Too many deep searches already running. "
                                     "Wait for one to finish or cancel it."}), 429
        _ultra_jobs[job_id] = {"state": "running", "checked": 0, "total": 0,
                               "best": None, "result": None, "error": None,
                               "cancel": False, "started": time.time(),
                               "ended": 0, "stage": "", "live_result": None}
    threading.Thread(target=_run_ultra, args=(job_id, body, world),
                     daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/ultra/<job_id>")
def ultra_status(job_id):
    job = _ultra_jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown or expired search."}), 404
    out = {"state": job["state"], "checked": job["checked"],
           "total": job["total"], "best": job["best"],
           "stage": job.get("stage", ""),
           "elapsed": round(time.time() - job["started"], 1)}
    if job["state"] in ("done", "cancelled") and job["result"]:
        out["result"] = job["result"]
    if job["error"]:
        out["error"] = job["error"]
    return jsonify(out)


@app.route("/api/ultra/<job_id>/keep", methods=["POST"])
def ultra_keep(job_id):
    """Stop the search and hand back the best plan found so far, immediately."""
    job = _ultra_jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown or expired search."}), 404

    # Check BEFORE cancelling. Setting the flag first meant an early Keep killed
    # the search and then reported "nothing found yet" - the job was dead but
    # the page carried on showing a running timer that could never finish.
    result = job.get("result") or job.get("live_result")
    if result is None:
        return jsonify({"error": "Nothing found yet - give it a moment."}), 409

    job["cancel"] = True
    return jsonify({"state": "kept", "checked": job["checked"],
                    "total": job["total"], "result": result})


@app.route("/api/ultra/<job_id>/cancel", methods=["POST"])
def ultra_cancel(job_id):
    job = _ultra_jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown or expired search."}), 404
    job["cancel"] = True
    return jsonify({"state": job["state"], "cancelling": True})


def _serialisable_layers(chain):
    return [[n["id"] for n in layer] for layer in chain["layers"]]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(HERE, "main.html")


# ---------------------------------------------------------------------------
# Share-link solver (unchanged behaviour)
# ---------------------------------------------------------------------------

@app.route("/api/solve", methods=["POST"])
def solve():
    body = request.get_json(silent=True)
    if not body or "share" not in body:
        return jsonify({"error": "Missing 'share' field in request body."}), 400

    share = body["share"].strip()
    if not share:
        return jsonify({"error": "Share code cannot be empty."}), 400

    try:
        chain = get_production_chain(share)
    except Exception as e:                                # noqa: BLE001
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "meta":   chain["meta"],
        "nodes":  chain["nodes"],
        "edges":  chain["edges"],
        "layers": _serialisable_layers(chain),
    })


# ---------------------------------------------------------------------------
# Reference data for the calculator UI
# ---------------------------------------------------------------------------

@app.route("/api/reference")
def reference():
    """Slimmed game data so the client can render pickers and toggles."""
    if calculator is None:
        return jsonify({"error": f"Calculator unavailable: {CALC_ERROR}"}), 503
    data = load_docs()

    # Only items that some machine recipe can actually produce are pickable.
    producible = set()
    for r in data["recipes"].values():
        for p in r["products"]:
            producible.add(p["item"])

    items = [{
        "id":    cn,
        "name":  it["name"],
        "fluid": it.get("fluid", False),
        "raw":   it.get("resource", False),
    } for cn, it in data["items"].items() if cn in producible or it.get("resource")]
    items.sort(key=lambda x: x["name"])

    recipes = [{
        "id":        cn,
        "name":      r["name"],
        "alternate": r["alternate"],
        "machine":   (r["producedIn"] or [None])[0],
        "products":  [p["item"] for p in r["products"]],
    } for cn, r in data["recipes"].items()]
    recipes.sort(key=lambda x: x["name"])

    machines = [{
        "id":    cn,
        "name":  b["name"],
        "power": b.get("power", 0.0),
    } for cn, b in data["buildings"].items()]
    machines.sort(key=lambda x: x["name"])

    resources = [{
        "id":        cn,
        "name":      info["name"],
        "max":       None if info["unlimited"] else info["max"],   # null = unlimited
        "unlimited": info["unlimited"],
    } for cn, info in data["resources"].items()
        if info["unlimited"] or info["max"] > 0]
    resources.sort(key=lambda x: x["name"])

    return jsonify({"items": items, "recipes": recipes,
                    "machines": machines, "resources": resources})


# ---------------------------------------------------------------------------
# World import
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# World import - also a background job. A late-game save is ~15MB and takes the
# better part of a minute to decode, which is far longer than a browser will sit
# on a single request; doing it inline is what made imports look flaky.
# ---------------------------------------------------------------------------

_world_jobs = {}
_world_jobs_lock = threading.Lock()


def _world_prune():
    now = time.time()
    for jid, job in list(_world_jobs.items()):
        if job["state"] in ("done", "error") and now - job["ended"] > 900:
            _world_jobs.pop(jid, None)


def _run_world(job_id, path, original, cleanup):
    job = _world_jobs[job_id]

    def progress(stage, frac):
        job["stage"] = stage
        job["percent"] = round(float(frac) * 100, 1)

    try:
        # Out of process: the decoder is pure Python and would otherwise hold the
        # GIL for the whole parse, hanging every other request on the server.
        from save_parser import start_parse_process
        proc, queue = start_parse_process(path)
        world = None
        while True:
            if not proc.is_alive() and queue.empty():
                break
            try:
                kind, *payload = queue.get(timeout=0.5)
            except Exception:                         # noqa: BLE001
                continue
            if kind == "progress":
                progress(payload[0], payload[1])
            elif kind == "done":
                world = payload[0]
                break
            elif kind == "error":
                raise RuntimeError(payload[0])
        proc.join(timeout=5)
        if world is None:
            raise RuntimeError("The save decoder stopped unexpectedly.")

        world_id = hashlib.sha1(
            f"{original}{len(world['recipes'])}{time.time()}".encode()).hexdigest()[:12]
        with _worlds_lock:
            _worlds[world_id] = world
        job["result"] = {
            "world_id":   world_id,
            "name":       os.path.splitext(original)[0] or "World",
            "parser":     world["parser"],
            "schematics": len(world["schematics"]),
            "recipes":    world["recipes"],
            "machines":   world["machines"],
            "resources":  world["resources"],
            "unknown":    world["unknown_schematics"],
        }
        job["state"] = "done"
        job["percent"] = 100.0
        job["stage"] = "Done"
    except Exception as e:                                # noqa: BLE001
        job["state"] = "error"
        job["error"] = f"Could not read save: {e}"
    finally:
        job["ended"] = time.time()
        if cleanup:
            try:
                os.unlink(path)
            except OSError:
                pass


@app.route("/api/world", methods=["POST"])
def world_upload():
    """Accept a .sav upload and start decoding it in the background."""
    if parse_save is None:
        return jsonify({"error": f"Save import unavailable: {SAVE_ERROR}"}), 503

    f = request.files.get("save")
    cleanup = False
    if f is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".sav", delete=False)
        f.save(tmp.name)
        tmp.close()
        path, original, cleanup = tmp.name, (f.filename or "world.sav"), True
    else:
        body = request.get_json(silent=True) or {}
        path = (body.get("path") or "").strip()
        original = os.path.basename(path) if path else ""
        if not path:
            return jsonify({"error": "Upload a .sav file or supply a path."}), 400
        if not os.path.exists(path):
            return jsonify({"error": f"Save file not found: {path}"}), 400

    job_id = hashlib.sha1(f"{time.time()}{original}".encode()).hexdigest()[:12]
    with _world_jobs_lock:
        _world_prune()
        _world_jobs[job_id] = {"state": "running", "percent": 0.0,
                               "stage": "Starting", "result": None,
                               "error": None, "started": time.time(), "ended": 0}
    threading.Thread(target=_run_world, args=(job_id, path, original, cleanup),
                     daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/world/<job_id>")
def world_status(job_id):
    job = _world_jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown or expired import."}), 404
    out = {"state": job["state"], "percent": job["percent"],
           "stage": job["stage"],
           "elapsed": round(time.time() - job["started"], 1)}
    if job["result"]:
        out["result"] = job["result"]
    if job["error"]:
        out["error"] = job["error"]
    return jsonify(out)


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

@app.route("/api/calculate", methods=["POST"])
def calculate_route():
    if calculator is None:
        return jsonify({"error": f"Calculator unavailable: {CALC_ERROR}"}), 503

    body = request.get_json(silent=True) or {}
    targets = body.get("targets") or []
    if not targets:
        return jsonify({"error": "Add at least one target item."}), 400

    mode = body.get("mode", "least_power")
    depth = body.get("depth", "quick")
    world_id = body.get("world_id")

    world = None
    if world_id:
        with _worlds_lock:
            world = _worlds.get(world_id)
        if world is None:
            return jsonify({"error": "World not loaded. Re-import your save."}), 400

    key = _cache_key(body)
    # Pressing Calculate should recompute from scratch rather than hand back
    # whatever the live preview happened to leave in the cache.
    cached = None if body.get("no_cache") else _calc_cache.get(key)
    if cached is not None:
        out = dict(cached)
        out["cached"] = True
        return jsonify(out)

    try:
        chain = calculator.calculate(
            targets, mode=mode, depth=depth, world=world,
            blocked_recipes=body.get("blocked_recipes") or (),
            blocked_machines=body.get("blocked_machines") or (),
            name=body.get("name"),
            resource_limits=body.get("resource_limits") or None,
            least_resources=body.get("least_resources") or "off",
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:                                # noqa: BLE001
        return jsonify({"error": f"Calculation failed: {e}"}), 500

    payload = {
        "meta":   chain["meta"],
        "nodes":  chain["nodes"],
        "edges":  chain["edges"],
        "layers": _serialisable_layers(chain),
        "stats":  chain["stats"],
        "cached": False,
    }
    _cache_store(key, payload)
    return jsonify(payload)


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "true").lower() == "true"

    print("[INFO] Starting Satisfactory Production Chain Helper")
    print(f"[INFO] Open http://localhost:{port} in your browser")
    print(f"[INFO] Debug mode: {debug}")

    app.run(host="0.0.0.0", port=port, debug=debug)
