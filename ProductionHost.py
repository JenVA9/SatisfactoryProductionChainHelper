# -*- coding: utf-8 -*-
"""
ProductionHost.py
Flask server for the Satisfactory Production Chain Helper.
Serves main.html and exposes the solver API.
"""

import os
import sys
from flask import Flask, send_from_directory

# ---------------------------------------------------------------------------
# Sanity check - make sure ShareCodeResolver is available before we start
# ---------------------------------------------------------------------------

if not os.path.exists(os.path.join(os.path.dirname(__file__), "ShareCodeResolver.py")):
    print("[ERROR] ShareCodeResolver.py not found in the same directory.")
    print("        Make sure both files are in the same folder before running.")
    sys.exit(1)

try:
    from ShareCodeResolver import get_production_chain
except ImportError as e:
    print(f"[ERROR] Could not import ShareCodeResolver: {e}")
    print("        Run: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(
    __name__,
    static_folder=os.path.dirname(__file__),   # serve files from same directory
)

# ---------------------------------------------------------------------------
# Block 2 - Serve main.html
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "main.html")

# ---------------------------------------------------------------------------
# Block 3 - Solve endpoint
# ---------------------------------------------------------------------------

from flask import request, jsonify

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
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # layers contains node objects - send just IDs, frontend reconstructs from nodes list
    serialisable_layers = [
        [node["id"] for node in layer]
        for layer in chain["layers"]
    ]

    return jsonify({
        "meta":   chain["meta"],
        "nodes":  chain["nodes"],
        "edges":  chain["edges"],
        "layers": serialisable_layers,
    })

# ---------------------------------------------------------------------------
# Block 4 - Run config
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "true").lower() == "true"

    print("[INFO] Starting Satisfactory Production Chain Helper")
    print(f"[INFO] Open http://localhost:{port} in your browser")
    print(f"[INFO] Debug mode: {debug}")

    app.run(host="0.0.0.0", port=port, debug=debug)