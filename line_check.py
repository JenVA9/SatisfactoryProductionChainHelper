# -*- coding: utf-8 -*-
"""
Check whether a solidified production line still matches the game data.

A solidified line stores the finished chain rather than the inputs that
produced it, so it renders identically for ever. What it cannot promise is that
the *game* still agrees: a recipe can be rebalanced, renamed away, or dropped.
This module answers one question - does every machine in this line still do
what the line says it does?

Two kinds of problem, kept deliberately separate:

  "broken"  - the file itself does not hold together (missing fields, edges
              pointing at nodes that are not there). Refuse it; it was not
              produced by this app, or it was edited by hand.
  "drift"   - the file is fine but the game moved under it. Import it and warn,
              because the line is still a perfectly good record of what someone
              built - it just no longer matches this version.

The amounts are re-derived with `_items_per_min`, the same function that
produced them, so a line that has not drifted compares exactly. The tolerance
below exists only for float round-tripping through JSON; a real recipe change
moves numbers by whole percent, which is millions of times larger.
"""
from docs_parser import load_docs
from ShareCodeResolver import _items_per_min, resolve_machine

FMT = "satisfactory-line"
VERSION = 1

# Relative tolerance for "these two rates are the same number". Generous enough
# that JSON float round-tripping can never trip it, tight enough that any real
# recipe change is caught.
REL_TOL = 1e-6
ABS_TOL = 1e-9

NON_RECIPE_KINDS = ("raw", "product", "byproduct", "input", "sink")

# The share resolver emits pseudo-nodes that are not game items at all.
SENTINELS = ("special__",)


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= max(ABS_TOL, REL_TOL * max(abs(a), abs(b)))


def _check_structure(doc) -> list:
    """Is this actually one of our files, and internally consistent?"""
    broken = []
    if not isinstance(doc, dict):
        return ["Not a production line file."]
    if doc.get("fmt") != FMT:
        return ["Not a solidified production line file."]
    v = doc.get("v")
    if not isinstance(v, int) or v < 1:
        return ["Missing or invalid format version."]
    if v > VERSION:
        return [f"This line was saved by a newer version of the app "
                f"(format {v}, this build reads {VERSION})."]

    chain = doc.get("chain")
    if not isinstance(chain, dict):
        return ["The file has no chain in it."]
    nodes = chain.get("nodes")
    edges = chain.get("edges")
    layers = chain.get("layers")
    if not isinstance(nodes, list) or not nodes:
        return ["The line has no machines in it."]
    if not isinstance(edges, list) or not isinstance(layers, list):
        return ["The line is missing its connections or layout."]

    ids = set()
    for n in nodes:
        if not isinstance(n, dict) or "id" not in n or "kind" not in n:
            broken.append("A node is missing its id or kind.")
            continue
        if n["id"] in ids:
            broken.append(f"Two nodes share the id {n['id']}.")
        ids.add(n["id"])

    for e in edges:
        if not isinstance(e, dict) or "from_id" not in e or "to_id" not in e:
            broken.append("A connection is missing one of its ends.")
            continue
        if e["from_id"] not in ids or e["to_id"] not in ids:
            broken.append("A connection points at a machine that is not in the file.")

    # Layers are id lists in a saved file, but a chain straight out of the
    # solver still has whole node dicts in them - accept either.
    flat = []
    for layer in layers:
        if not isinstance(layer, list):
            continue
        for entry in layer:
            flat.append(entry.get("id") if isinstance(entry, dict) else entry)
    if len(flat) != len(set(flat)) or set(flat) != ids:
        broken.append("The layout does not cover every machine exactly once.")

    # Only ever report each distinct complaint once.
    seen, unique = set(), []
    for b in broken:
        if b not in seen:
            seen.add(b)
            unique.append(b)
    return unique


def check(doc, data=None) -> dict:
    """
    Returns {"ok", "broken": [...], "drift": [...], "summary": str}.

    `ok` is True only when the file holds together AND still matches the game.
    """
    broken = _check_structure(doc)
    if broken:
        return {"ok": False, "broken": broken, "drift": [],
                "summary": "This file could not be read."}

    data = data or load_docs()
    recipes = data["recipes"]
    buildings = data["buildings"]
    items = data["items"]

    drift = []
    checked = 0

    for n in doc["chain"]["nodes"]:
        kind = n.get("kind")

        if kind in NON_RECIPE_KINDS:
            it = n.get("item")
            if it and any(it.startswith(x) for x in SENTINELS):
                continue
            if it and it not in items:
                drift.append(f"{n.get('label') or it} no longer exists in the game.")
            continue

        if kind != "recipe":
            continue

        rc = n.get("recipe")
        mc = n.get("machine")
        label = n.get("label") or rc or "a machine"

        if rc and any(rc.startswith(x) for x in SENTINELS):
            continue

        recipe = recipes.get(rc)
        if recipe is None:
            drift.append(f"The recipe \"{label}\" no longer exists.")
            continue
        # Does the machine still exist under either prefix? Resolution is what
        # makes this safe: a share-link chain names it `Desc_*` and the table is
        # keyed `Build_*`, so a plain lookup used to condemn every machine in an
        # imported line. Only a class that resolves to nothing is really gone -
        # and a line calling for a machine that no longer exists is unbuildable
        # however well its numbers add up.
        machine = resolve_machine(buildings, mc)
        if mc and not machine:
            drift.append(f"\"{label}\" is built in a machine that no longer "
                         f"exists ({n.get('machine_label') or mc}).")
            continue

        try:
            ing, prod = _items_per_min(recipe, machine,
                                       int(n.get("clock", 100) or 100),
                                       float(n.get("count", 0.0)))
        except Exception:                                 # noqa: BLE001
            drift.append(f"\"{label}\" could not be re-checked against the "
                         f"current game data.")
            continue

        checked += 1
        for side, now, was in (("takes", ing, n.get("inputs") or {}),
                               ("makes", prod, n.get("outputs") or {})):
            if set(now) != set(was):
                added = sorted(set(now) - set(was))
                gone = sorted(set(was) - set(now))
                bits = []
                if added:
                    bits.append("now " + side + " " +
                                ", ".join(_name(items, x) for x in added))
                if gone:
                    bits.append("no longer " + side + " " +
                                ", ".join(_name(items, x) for x in gone))
                drift.append(f"\"{label}\" {'; '.join(bits)}.")
                continue
            for k, v in now.items():
                if not _same(float(v), float(was.get(k, 0.0))):
                    drift.append(
                        f"\"{label}\" {side} {_name(items, k)} at "
                        f"{float(was.get(k, 0.0)):.4g}/min in this line, but "
                        f"{float(v):.4g}/min in the game now.")

    # Same complaint can arise from several identical machines - say it once.
    seen, unique = set(), []
    for d in drift:
        if d not in seen:
            seen.add(d)
            unique.append(d)

    ok = not unique
    return {
        "ok": ok,
        "broken": [],
        "drift": unique,
        "checked": checked,
        "summary": (f"Verified against the current game data - all {checked} "
                    f"machines still match."
                    if ok else
                    f"{len(unique)} thing(s) in this line no longer match the "
                    f"current game data."),
    }


def _name(items: dict, cls: str) -> str:
    return (items.get(cls) or {}).get("name") or cls


def build(chain: dict, stats: dict, name: str, source: str = "calc") -> dict:
    """Wrap a finished chain as a solidified line file."""
    import datetime
    return {
        "fmt": FMT,
        "v": VERSION,
        "name": name or "Production line",
        "solidified": datetime.datetime.now(datetime.timezone.utc)
                              .replace(microsecond=0).isoformat(),
        "source": source,
        "chain": {
            "meta":   chain.get("meta") or {},
            "nodes":  chain.get("nodes") or [],
            "edges":  chain.get("edges") or [],
            # Always stored as id lists, whichever shape came in - the file is
            # what gets shared, so it should have exactly one layout format.
            "layers": [
                [(e.get("id") if isinstance(e, dict) else e) for e in layer]
                for layer in (chain.get("layers") or [])
            ],
        },
        "stats": stats or {},
    }
