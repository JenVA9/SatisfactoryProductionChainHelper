# -*- coding: utf-8 -*-
"""
ProductionHost.py
Flask server for the Satisfactory Production Chain Helper.
Serves main.html, the share-link solver, and the production calculator.
"""

import os
import sys
import json
import time
import hashlib
import subprocess
import tempfile
import threading

import job_store

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
# Result cache. Parsed worlds live in job_store, which every worker can read.
# ---------------------------------------------------------------------------

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
        # Atomic: several workers share this file, and a plain truncate-then-
        # write leaves a half-file behind if two land together - which then
        # fails to parse on the next start and the whole cache is lost.
        try:
            fd, tmp = tempfile.mkstemp(dir=HERE, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(_calc_cache, f)
            os.replace(tmp, CACHE_PATH)
        except Exception:                                 # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# ULTRA jobs - long searches run in the background so the request can return
# and the browser can poll for progress.
# ---------------------------------------------------------------------------

_ULTRA_MAX_RUNNING = 3
_STALE_AFTER = 90          # no heartbeat for this long => the child is gone


def _spawn_runner(module, job_id):
    """
    Start a runner as a plain subprocess.

    Not multiprocessing: under gunicorn `__main__` is the gunicorn console
    script, and the spawn start method re-imports `__main__` in the child,
    which would try to boot a second gunicorn. `-m` has no such fixup.
    """
    log = open(os.path.join(job_store.job_dir(job_id), "stderr.log"), "wb")
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        # Its own session, so a worker reload or Ctrl-C does not take the
        # search down with it.
        kw["start_new_session"] = True
    return subprocess.Popen(
        [sys.executable, "-m", module, job_id],
        cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=log, **kw)


def _job_view(job_id, running_states=("running",)):
    """Read a job's status, deciding whether a quiet child has died."""
    st = job_store.read_json(job_id, "status")
    if st is None:
        return None
    if st.get("state") in running_states:
        quiet = time.time() - float(st.get("alive") or st.get("started") or 0)
        if quiet > _STALE_AFTER:
            st["state"] = "error"
            st.setdefault("error", "The search stopped unexpectedly.")
    return st


def _running_ultras():
    try:
        ids = os.listdir(job_store.JOBS_ROOT)
    except OSError:
        return 0
    n = 0
    for jid in ids:
        if jid == "worlds" or not jid.isalnum():
            continue
        st = job_store.read_json(jid, "status")
        if st and st.get("state") == "running" and st.get("kind") == "ultra":
            n += 1
    return n


@app.route("/api/ultra", methods=["POST"])
def ultra_start():
    if calculator is None:
        return jsonify({"error": f"Calculator unavailable: {CALC_ERROR}"}), 503
    body = request.get_json(silent=True) or {}
    if not (body.get("targets") or []):
        return jsonify({"error": "Add at least one target item."}), 400

    if body.get("world_id") and job_store.get_world(body["world_id"]) is None:
        return jsonify({"error": "World not loaded. Re-import your save."}), 400

    job_store.prune()
    if _running_ultras() >= _ULTRA_MAX_RUNNING:
        return jsonify({"error": "Too many deep searches already running. "
                                 "Wait for one to finish or cancel it."}), 429

    # Share the machine between searches instead of each sizing its own pool
    # against the whole box.
    running = max(1, _running_ultras() + 1)
    workers = max(2, min(8, calculator.ULTRA_MAX_CONCURRENT_SOLVES // running))

    job_id = job_store.new_id()
    try:
        job_store.create(job_id)
    except OSError as e:
        # Deployed somewhere the app directory is not writable. Say so plainly -
        # JOBS_DIR moves the store somewhere that is.
        return jsonify({"error": f"Cannot write job state to "
                                 f"{job_store.JOBS_ROOT}: {e}. "
                                 f"Set JOBS_DIR to a writable directory."}), 500
    job_store.write_json(job_id, "payload", {
        "targets":          body.get("targets") or [],
        "mode":             body.get("mode", "least_machines"),
        "world_id":         body.get("world_id"),
        "blocked_recipes":  body.get("blocked_recipes") or [],
        "blocked_machines": body.get("blocked_machines") or [],
        "resource_limits":  body.get("resource_limits") or None,
        "least_resources":  body.get("least_resources") or "off",
        "name":             body.get("name"),
        "priority":         body.get("priority") or None,
        "ultra_depth":      body.get("ultra_depth"),   # NOT "depth"
        "workers":          workers,
        "started":          time.time(),
    })
    # Written before the child starts: it takes a few seconds to import SciPy
    # and parse the docs, and a poll in that window must not 404.
    job_store.write_json(job_id, "status", {
        "kind": "ultra", "state": "running", "checked": 0, "total": 0,
        "stage": "Starting", "best": None, "error": None,
        "started": time.time(), "alive": time.time()})

    try:
        _spawn_runner("ultra_runner", job_id)
    except Exception as e:                                # noqa: BLE001
        job_store.write_json(job_id, "status", {
            "kind": "ultra", "state": "error", "checked": 0, "total": 0,
            "stage": "", "best": None, "error": f"Could not start search: {e}",
            "started": time.time(), "alive": time.time()})
        return jsonify({"error": f"Could not start search: {e}"}), 500

    return jsonify({"job_id": job_id})


@app.route("/api/ultra/<job_id>")
def ultra_status(job_id):
    st = _job_view(job_id)
    if st is None:
        return jsonify({"error": "Unknown or expired search."}), 404
    out = {"state": st.get("state"), "checked": st.get("checked", 0),
           "total": st.get("total", 0), "best": st.get("best"),
           "stage": st.get("stage", ""),
           "elapsed": round(time.time() - float(st.get("started") or time.time()), 1)}
    if st.get("state") in ("done", "cancelled"):
        result = job_store.read_json(job_id, "result") \
            or job_store.read_json(job_id, "live")
        if result:
            out["result"] = result
    if st.get("error"):
        out["error"] = st["error"]
    return jsonify(out)


@app.route("/api/ultra/<job_id>/keep", methods=["POST"])
def ultra_keep(job_id):
    """Stop the search and hand back the best plan found so far, immediately."""
    st = _job_view(job_id)
    if st is None:
        return jsonify({"error": "Unknown or expired search."}), 404

    # Check BEFORE stopping. Setting the flag first meant an early Keep killed
    # the search and then reported "nothing found yet" - the job was dead but
    # the page carried on showing a running timer that could never finish.
    result = job_store.read_json(job_id, "result") \
        or job_store.read_json(job_id, "live")
    if result is None:
        return jsonify({"error": "Nothing found yet - give it a moment."}), 409

    job_store.request_stop(job_id)
    return jsonify({"state": "kept", "checked": st.get("checked", 0),
                    "total": st.get("total", 0), "result": result})


@app.route("/api/ultra/<job_id>/cancel", methods=["POST"])
def ultra_cancel(job_id):
    if not job_store.exists(job_id):
        return jsonify({"error": "Unknown or expired search."}), 404
    job_store.request_stop(job_id)
    st = job_store.read_json(job_id, "status") or {}
    return jsonify({"state": st.get("state", "running"), "cancelling": True})


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

    job_store.prune()
    job_id = job_store.new_id()
    try:
        job_store.create(job_id)
    except OSError as e:
        return jsonify({"error": f"Cannot write job state to "
                                 f"{job_store.JOBS_ROOT}: {e}. "
                                 f"Set JOBS_DIR to a writable directory."}), 500
    job_store.write_json(job_id, "payload", {
        "path": path, "original": original, "cleanup": cleanup,
        "started": time.time()})
    job_store.write_json(job_id, "status", {
        "kind": "world", "state": "running", "percent": 0.0,
        "stage": "Starting", "error": None,
        "started": time.time(), "alive": time.time()})

    try:
        _spawn_runner("world_runner", job_id)
    except Exception as e:                                # noqa: BLE001
        return jsonify({"error": f"Could not start import: {e}"}), 500

    return jsonify({"job_id": job_id})


@app.route("/api/world/<job_id>")
def world_status(job_id):
    st = _job_view(job_id)
    if st is None:
        return jsonify({"error": "Unknown or expired import."}), 404
    out = {"state": st.get("state"), "percent": st.get("percent", 0.0),
           "stage": st.get("stage", ""),
           "elapsed": round(time.time() - float(st.get("started") or time.time()), 1)}
    result = job_store.read_json(job_id, "result")
    if result:
        out["result"] = result
    if st.get("error"):
        out["error"] = st["error"]
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
        world = job_store.get_world(world_id)
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
