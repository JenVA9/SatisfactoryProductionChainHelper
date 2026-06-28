# -*- coding: utf-8 -*-
"""
satisfactory_share.py

Library for fetching and solving satisfactorytools.com share links,
including full production graph with connections.

Public API
----------
get_production_chain(share_code_or_url, game_version=None)
    -> dict with keys: meta, nodes, edges, layers

nodes  : list of node dicts (one per production step / raw resource / output)
edges  : list of edge dicts  {from_id, to_id, item, amount}  (items/min)
layers : nodes grouped in dependency order (layer 0 = no dependencies, last = outputs)

Node dict keys:
    id          : unique int
    kind        : "recipe" | "raw" | "product" | "byproduct" | "input" | "sink"
    recipe      : recipe className  (kind=="recipe" only)
    machine     : machine className (kind=="recipe" only)
    clock       : clock speed %     (kind=="recipe" only)
    count       : machine count     (kind=="recipe" only)
    item        : item className    (non-recipe kinds)
    label       : human-readable name
    inputs      : {item_cls: amount/min}   what this node consumes
    outputs     : {item_cls: amount/min}   what this node produces
"""

import json
import math
import requests
import urllib.request
from urllib.parse import urlparse, parse_qs

try:
    from docs_parser import load_docs as _load_docs_local
except ImportError:
    _load_docs_local = None

SHARE_API   = "https://api.satisfactorytools.com/v2/share"
SOLVER_API  = "https://api.satisfactorytools.com/v2/solver"
_HEADERS = {
    "Content-Type": "application/json",
    "Accept":       "application/json",
    "Origin":       "https://www.satisfactorytools.com",
    "Referer":      "https://www.satisfactorytools.com/",
}
_VERSION_MAP = {
    "1.0":         "1.0.0",
    "1.0-ficsmas": "1.0.0-ficsmas",
    "0.8":         "0.8.0",
}

# Docs JSON path - set this to your local Satisfactory install.
# If None, docs_parser will search common Steam locations automatically.
DOCS_JSON_PATH = None

# Fallback online data source (missing some 1.0 recipes)
DATA_URL = "https://raw.githubusercontent.com/greeny/SatisfactoryTools/master/data/data.json"

# Cache
_game_data: dict | None = None


def _load_game_data() -> dict:
    global _game_data
    if _game_data is not None:
        return _game_data

    # Try local Docs JSON first (complete, always up to date with the game)
    if _load_docs_local is not None:
        try:
            _game_data = _load_docs_local(DOCS_JSON_PATH)
            return _game_data
        except Exception as e:
            print(f"[warn] Could not load local Docs JSON: {e}. Falling back to online data.")

    # Fallback: fetch from GitHub (may be missing some recipes)
    req = urllib.request.Request(DATA_URL, headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.satisfactorytools.com/",
    })
    with urllib.request.urlopen(req, timeout=15) as r:
        _game_data = json.loads(r.read())
    return _game_data


def _item_name(cls: str, data: dict) -> str:
    return data["items"].get(cls, {}).get("name", cls)


def _recipe_name(cls: str, data: dict) -> str:
    return data["recipes"].get(cls, {}).get("name", cls)


def _machine_name(cls: str, data: dict) -> str:
    return data["buildings"].get(cls, {}).get("name", cls)


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def _extract_code(arg: str) -> tuple:
    if arg.startswith("http"):
        parsed = urlparse(arg)
        parts = [p for p in parsed.path.split("/") if p]
        url_version = parts[0] if parts else "1.0"
        qs = parse_qs(parsed.query)
        if "share" not in qs:
            raise ValueError(f"No ?share= in URL: {arg}")
        return qs["share"][0], url_version
    return arg.strip(), "1.0"


def _fetch_share(code: str) -> dict:
    r = requests.get(f"{SHARE_API}/{code}", headers=_HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()["data"]


def _solve(request: dict, game_version: str) -> dict:
    payload = dict(request)
    payload["gameVersion"] = game_version
    payload.pop("blockedMachines", None)
    r = requests.post(SOLVER_API, json=payload, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json().get("result", {})


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

def _items_per_min(recipe: dict, machine: dict, clock: int, machine_count: float) -> dict:
    """
    Compute actual items/min for each ingredient and product given
    machine count, clock speed, and recipe data.
    """
    mfg_speed  = machine.get("metadata", {}).get("manufacturingSpeed", 1.0)
    cycle_time  = recipe["time"]  # seconds per cycle at 100% clock
    # per-machine rate at given clock:  (amount/cycle) * (60/cycle_time) * mfgSpeed * (clock/100)
    multiplier = (60.0 / cycle_time) * mfg_speed * (clock / 100.0) * machine_count
    result = {}
    for entry in recipe.get("ingredients", []):
        result[entry["item"]] = entry["amount"] * multiplier
    return result, {
        entry["item"]: entry["amount"] * multiplier
        for entry in recipe.get("products", [])
    }


def _build_graph(solver_result: dict, data: dict) -> tuple:
    """
    Returns (nodes, edges) where nodes is a list of node dicts
    and edges is a list of {from_id, to_id, item, amount}.
    """
    DELTA = 1e-6
    nodes = []
    next_id = [1]

    def make_id():
        i = next_id[0]
        next_id[0] += 1
        return i

    # --- Parse solver result into node dicts ---
    for key, raw_val in solver_result.items():
        value = float(raw_val)
        if "#" not in key:
            continue
        left, kind = key.split("#", 1)

        if kind in ("Mine", "Product", "Byproduct", "Input", "Sink"):
            item_cls = left
            node_kind = {
                "Mine":      "raw",
                "Product":   "product",
                "Byproduct": "byproduct",
                "Input":     "input",
                "Sink":      "sink",
            }[kind]
            nodes.append({
                "id":      make_id(),
                "kind":    node_kind,
                "item":    item_cls,
                "label":   _item_name(item_cls, data),
                "amount":  value,
                "inputs":  {},
                "outputs": {item_cls: value} if node_kind in ("raw", "input") else {},
                "outputs_remaining": {item_cls: value} if node_kind in ("raw", "input") else {},
                "inputs_remaining":  {item_cls: value} if node_kind in ("product", "sink", "byproduct") else {},
            })
        else:
            # Recipe node: left = "recipeClass@clock", kind = machineClass
            machine_cls = kind
            if "@" in left:
                recipe_cls, clock_str = left.split("@", 1)
                clock = int(clock_str)
            else:
                recipe_cls = left
                clock = 100

            if recipe_cls == "special__power":
                continue

            recipe  = data["recipes"].get(recipe_cls)
            machine = data["buildings"].get(machine_cls, {})

            if recipe is None:
                continue

            ing_rates, prod_rates = _items_per_min(recipe, machine, clock, value)

            nodes.append({
                "id":      make_id(),
                "kind":    "recipe",
                "recipe":  recipe_cls,
                "machine": machine_cls,
                "clock":   clock,
                "count":   value,
                "label":   _recipe_name(recipe_cls, data),
                "machine_label": _machine_name(machine_cls, data),
                "inputs":  ing_rates,
                "outputs": prod_rates,
                "outputs_remaining": dict(prod_rates),
                "inputs_remaining":  dict(ing_rates),
            })

    # --- Generate edges ---
    # Instead of iterating producers and greedily finding consumers (order-sensitive),
    # we iterate CONSUMERS and for each required input find the producer that makes it.
    # Terminal nodes (product/byproduct/sink) are handled last so they never steal
    # flow that a recipe needs.
    #
    # Build lookup: item_cls -> list of nodes that produce it
    producers_of: dict = {}
    for n in nodes:
        for item_cls in n["outputs_remaining"]:
            producers_of.setdefault(item_cls, []).append(n)

    edges = []

    def connect(consumer, item_cls, needed):
        """Draw as much of `needed` as possible from producers of item_cls."""
        for producer in producers_of.get(item_cls, []):
            if producer is consumer:
                continue
            available = producer["outputs_remaining"].get(item_cls, 0.0)
            if available <= DELTA:
                continue
            diff = min(needed, available)
            producer["outputs_remaining"][item_cls] -= diff
            consumer["inputs_remaining"][item_cls]  -= diff
            needed -= diff
            edges.append({
                "from_id":    producer["id"],
                "to_id":      consumer["id"],
                "item":       item_cls,
                "item_label": _item_name(item_cls, data),
                "amount":     diff,
            })
            if needed <= DELTA:
                break
        return needed

    # Pass 1: satisfy all recipe inputs first (prevents terminal nodes stealing flow)
    for node in nodes:
        if node["kind"] != "recipe":
            continue
        for item_cls, needed in list(node["inputs_remaining"].items()):
            if needed > DELTA:
                connect(node, item_cls, needed)

    # Pass 2: satisfy terminal node inputs (product/byproduct/sink)
    for node in nodes:
        if node["kind"] not in ("product", "byproduct", "sink"):
            continue
        for item_cls, needed in list(node["inputs_remaining"].items()):
            if needed > DELTA:
                connect(node, item_cls, needed)

    # Unsatisfied inputs (no producer in chain) are left without edges.
    # This is correct - the solver can return chains where some intermediate
    # items have no producer (e.g. the planner used a blocked recipe upstream).
    # We do not synthesise fake input nodes - the recipe simply has no incoming
    # edge for that item, which is accurate.

    # Clean up internal tracking fields
    for n in nodes:
        n.pop("outputs_remaining", None)
        n.pop("inputs_remaining", None)

    return nodes, edges


def _topological_layers(nodes: list, edges: list) -> list:
    """
    Kahn's algorithm with cycle breaking and late-placement of no-input recipe nodes.

    Recipe nodes with no incoming edges (e.g. Excited Photonic Matter — a converter
    with no ingredients) would normally land in layer 0 alongside raw resources.
    Instead we defer them: they are placed in the layer just before their first consumer,
    so they appear at the right point in the build order.

    Raw/input nodes with no incoming edges always stay in layer 0.
    """
    id_to_node = {n["id"]: n for n in nodes}
    in_edges  = {n["id"]: set() for n in nodes}
    out_edges = {n["id"]: set() for n in nodes}
    for e in edges:
        in_edges[e["to_id"]].add(e["from_id"])
        out_edges[e["from_id"]].add(e["to_id"])

    # Identify recipe nodes that have no inputs AND have at least one consumer.
    # These should be deferred to appear just before their consumer layer.
    def _should_defer(nid):
        node = id_to_node[nid]
        if node["kind"] != "recipe":
            return False
        if len(in_edges[nid]) > 0:
            return False   # has real deps, handle normally
        if len(out_edges[nid]) == 0:
            return False   # no consumers either, leave in layer 0
        return True

    deferred = {nid for nid in id_to_node if _should_defer(nid)}

    # Treat deferred nodes as if they depend on their consumers' dependencies
    # by temporarily giving them a virtual dependency that gets resolved when
    # their consumer is about to be placed.
    # We do this by simply excluding them from the initial zero-dep set and
    # instead injecting them into the layer immediately before their consumer.

    remaining_in = {nid: set(deps) for nid, deps in in_edges.items()}
    # For deferred nodes, add a placeholder so they don't appear in layer 0
    for nid in deferred:
        remaining_in[nid] = {"__deferred__"}

    layers = []
    processed = set()

    while len(processed) < len(nodes):
        # Before building the layer, inject any deferred nodes whose consumers
        # are now all in the ready set (i.e. consumers are about to be placed
        # or were already placed).
        for nid in list(deferred):
            if nid in processed:
                continue
            consumers = out_edges[nid]
            # Consumer is "ready to be placed next" if all ITS deps (other than
            # this deferred node) are already processed.
            consumer_about_to_run = any(
                all(dep == nid or dep in processed
                    for dep in in_edges[c])
                for c in consumers
            )
            if consumer_about_to_run:
                remaining_in[nid] = set()   # unlock it

        # Normal Kahn pass
        layer = [nid for nid, deps in remaining_in.items()
                 if len(deps) == 0 and nid not in processed]

        if not layer:
            # Cycle detected or all remaining are deferred — force unlock
            unprocessed = [nid for nid in remaining_in if nid not in processed]
            # Prefer deferred nodes that have consumers almost ready
            cycle_breaker = min(
                unprocessed,
                key=lambda nid: len(remaining_in[nid])
            )
            remaining_in[cycle_breaker] = set()
            layer = [cycle_breaker]

        layers.append([id_to_node[nid] for nid in layer])
        for nid in layer:
            processed.add(nid)
            remaining_in[nid] = set()
        for nid in layer:
            for consumer in out_edges[nid]:
                remaining_in[consumer].discard(nid)

    return layers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_production_chain(share_code_or_url: str, game_version: str = None) -> dict:
    """
    Fetch, solve, and build a full production graph from a satisfactorytools.com share link.

    Returns dict with:
        meta    : share metadata
        nodes   : all nodes (recipe steps, raw resources, products, byproducts, inputs, sinks)
        edges   : connections {from_id, to_id, item, item_label, amount}
        layers  : nodes in dependency order (list of lists)
    """
    code, url_version = _extract_code(share_code_or_url)
    if game_version is None:
        game_version = url_version
    api_version = _VERSION_MAP.get(game_version, "1.0.0")

    share_data   = _fetch_share(code)
    solver_result = _solve(share_data["request"], api_version)
    game_data    = _load_game_data()
    nodes, edges = _build_graph(solver_result, game_data)
    layers       = _topological_layers(nodes, edges)

    return {
        "meta":   share_data["metadata"],
        "nodes":  nodes,
        "edges":  edges,
        "layers": layers,
    }


# ---------------------------------------------------------------------------
# CLI pretty-print
# ---------------------------------------------------------------------------

def _fmt(n: float) -> str:
    s = f"{n:.4f}".rstrip("0").rstrip(".")
    return s


def _print_chain(chain: dict):
    nodes_by_id = {n["id"]: n for n in chain["nodes"]}
    # Build edge lookups
    edges_from = {}  # from_id -> [edge]
    edges_to   = {}  # to_id   -> [edge]
    for e in chain["edges"]:
        edges_from.setdefault(e["from_id"], []).append(e)
        edges_to.setdefault(e["to_id"],   []).append(e)

    meta = chain["meta"]
    print(f"\n{'='*70}")
    print(f"  {meta.get('name') or 'Unnamed'}")
    print(f"{'='*70}\n")

    for layer_idx, layer in enumerate(chain["layers"]):
        print(f"  ── LAYER {layer_idx} {'(raw inputs / no dependencies)' if layer_idx == 0 else ''}")
        for node in layer:
            nid   = node["id"]
            kind  = node["kind"]
            print()

            if kind == "recipe":
                whole = int(node["count"])
                extra = node["count"] - whole
                clock = node["clock"]
                clock_note = f" @{clock}%" if clock != 100 else ""
                if extra > 0.001:
                    count_str = f"{whole} + 1{clock_note} underclocked ({_fmt(extra*100)}%)"
                else:
                    count_str = f"{whole}{clock_note}"
                print(f"    [{nid:>3}] RECIPE  {node['label']}")
                print(f"          Machine : {node['machine_label']}  x{count_str}")

                # Where inputs come from
                if nid in edges_to:
                    print(f"          FROM:")
                    for e in edges_to[nid]:
                        src = nodes_by_id[e["from_id"]]
                        src_label = src.get("label","?")
                        print(f"            [{e['from_id']:>3}] {src_label:30s}  {_fmt(e['amount']):>10}/min  {e['item_label']}")

                # What this produces and where it goes
                if nid in edges_from:
                    print(f"          TO:")
                    for e in edges_from[nid]:
                        dst = nodes_by_id[e["to_id"]]
                        dst_label = dst.get("label","?")
                        print(f"            [{e['to_id']:>3}] {dst_label:30s}  {_fmt(e['amount']):>10}/min  {e['item_label']}")

            else:
                kind_label = {
                    "raw":       "RAW RESOURCE",
                    "product":   "OUTPUT",
                    "byproduct": "BYPRODUCT",
                    "input":     "MANUAL INPUT",
                    "sink":      "SINK",
                }.get(kind, kind.upper())
                print(f"    [{nid:>3}] {kind_label:14s}  {node['label']}  {_fmt(node['amount'])}/min")

                if nid in edges_from:
                    print(f"          TO:")
                    for e in edges_from[nid]:
                        dst = nodes_by_id[e["to_id"]]
                        print(f"            [{e['to_id']:>3}] {dst.get('label','?'):30s}  {_fmt(e['amount']):>10}/min")
                if nid in edges_to:
                    print(f"          FROM:")
                    for e in edges_to[nid]:
                        src = nodes_by_id[e["from_id"]]
                        print(f"            [{e['from_id']:>3}] {src.get('label','?'):30s}  {_fmt(e['amount']):>10}/min")

        print()

    print(f"  Total nodes : {len(chain['nodes'])}")
    print(f"  Total edges : {len(chain['edges'])}")
    print(f"  Layers      : {len(chain['layers'])}")
    print()


if __name__ == "__main__":
    import sys

    url = "https://www.satisfactorytools.com/1.0/production?share=pF5HeFjEUEjaaSPwoU9Q"
    if len(sys.argv) > 1:
        url = sys.argv[1]

    print(f"Fetching + solving: {url}")
    chain = get_production_chain(url)

    if "--json" in sys.argv:
        # Dump the raw structure (layers omitted as they reference node objects)
        print(json.dumps({
            "meta":  chain["meta"],
            "nodes": chain["nodes"],
            "edges": chain["edges"],
        }, indent=2))
    else:
        _print_chain(chain)