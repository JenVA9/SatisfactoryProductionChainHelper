# -*- coding: utf-8 -*-
"""
save_parser.py

Reads a Satisfactory .sav file and works out what the player has actually
unlocked, so the calculator can restrict itself to that playthrough.

Returns:
    {
      "schematics":       ["Schematic_1-1_C", ...],   # purchased, class names
      "recipes":          ["Recipe_IngotIron_C", ...],# unlocked, class names
      "machines":         ["Build_SmelterMk1_C", ...],# unlocked producers
      "resources":        { "Desc_OreIron_C": 92100.0, ... },
      "unknown_schematics": [...],   # modded / not in Docs, reported not fatal
      "parser":           "greyhak" | "fallback",
    }

Resource limits are map-wide totals (node occupancy is deliberately ignored);
water is unlimited. See docs_parser.RESOURCE_MAX.

Parsing strategy:
  1. GreyHak/sat_sav_parse (vendored, GPL-3) - complete and proven.
  2. A minimal built-in fallback that only pulls mPurchasedSchematics, used if
     the vendored parser is missing or fails.
"""

import os
import re
import sys
import zlib
import struct

from docs_parser import load_docs

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")


# ---------------------------------------------------------------------------
# Strategy 1: vendored GreyHak parser
# ---------------------------------------------------------------------------

def _schematics_via_greyhak(path: str) -> list:
    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)
    import sav_parse  # noqa: F401  (vendored)

    parsed = sav_parse.readFullSaveFile(path)

    # The persistent level is the one carrying essentially all the actors.
    level = max(parsed.levels,
                key=lambda lv: len(getattr(lv, "actorAndComponentObjectHeaders", []) or []))

    for obj in level.objects:
        # properties is a list of [name, value] pairs, not attribute objects.
        for prop in getattr(obj, "properties", []) or []:
            if len(prop) == 2 and prop[0] == "mPurchasedSchematics":
                return [_class_of(ref.pathName) for ref in (prop[1] or [])]
    return []


# ---------------------------------------------------------------------------
# Strategy 2: minimal fallback
# ---------------------------------------------------------------------------

_SCHEMATIC_RE = re.compile(rb"/Game/[A-Za-z0-9_/\-]*?Schematic[A-Za-z0-9_/\-]*?\.([A-Za-z0-9_\-]+_C)")


def _inflate_chunks(path: str) -> bytes:
    """
    Satisfactory saves are a plain header followed by zlib-compressed chunks.
    We do not need the header layout - we just walk the file looking for zlib
    streams and inflate everything we can. Crude, but it only has to be good
    enough to regex out schematic paths.
    """
    raw = open(path, "rb").read()
    out = bytearray()
    i = 0
    n = len(raw)
    while i < n - 1:
        # zlib streams start 0x78 with a valid FLG check byte
        if raw[i] == 0x78 and (((raw[i] << 8) | raw[i + 1]) % 31 == 0):
            try:
                d = zlib.decompressobj()
                out += d.decompress(raw[i:])
                i = n - len(d.unused_data)
                continue
            except zlib.error:
                pass
        i += 1
    return bytes(out)


def _schematics_via_fallback(path: str) -> list:
    blob = _inflate_chunks(path)
    if not blob:
        raise RuntimeError("Could not decompress any chunk from the save file.")
    # Deduplicate, preserving order.
    seen, out = set(), []
    for m in _SCHEMATIC_RE.finditer(blob):
        cn = m.group(1).decode("ascii", "ignore")
        if cn not in seen:
            seen.add(cn)
            out.append(cn)
    if not out:
        raise RuntimeError("No schematics found in save file.")
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_of(path_name: str) -> str:
    """'/Game/.../Schematic_1-1.Schematic_1-1_C' -> 'Schematic_1-1_C'"""
    return str(path_name).split(".")[-1]


def _machines_for(recipe_classes, data) -> list:
    machines, recipes = set(), data["recipes"]
    for rc in recipe_classes:
        for m in (recipes.get(rc) or {}).get("producedIn", []):
            machines.add(m)
    return sorted(machines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_save(path: str, docs_path: str = None, progress=None) -> dict:
    """
    Read a .sav and return the unlocked recipe/machine/resource picture.

    `progress(stage, fraction)` is called as the work moves along. A late-game
    save is ~15MB and takes the better part of a minute to decode, which is far
    longer than a browser will sit on a request - so the caller needs to be able
    to show what is happening rather than appearing to hang.
    """
    def say(stage, frac):
        if progress:
            try:
                progress(stage, frac)
            except Exception:                       # noqa: BLE001
                pass

    if not os.path.exists(path):
        raise FileNotFoundError(f"Save file not found: {path}")

    say("Reading save file", 0.05)
    parser_used = "greyhak"
    try:
        say("Decoding save (this is the slow part)", 0.15)
        schematics = _schematics_via_greyhak(path)
        if not schematics:
            raise RuntimeError("no mPurchasedSchematics found")
    except Exception as exc:                        # noqa: BLE001
        print(f"[save_parser] Vendored parser unavailable/failed ({exc}); "
              f"using fallback.")
        schematics = _schematics_via_fallback(path)
        parser_used = "fallback"

    say("Loading game data", 0.75)
    data = load_docs(docs_path)
    known = data["schematics"]

    say("Matching unlocked schematics", 0.85)

    unlocked, unknown = set(), []
    for sc in schematics:
        entry = known.get(sc)
        if entry is None:
            # Modded schematics (e.g. SnappableExtractors) are not in the
            # vanilla Docs - skip them rather than failing the import.
            unknown.append(sc)
            continue
        unlocked.update(entry["recipes"])

    # Only keep recipes we actually have machine data for.
    recipes = sorted(r for r in unlocked if r in data["recipes"])

    say("Working out available recipes", 0.95)
    resources = {
        cn: (None if info["unlimited"] else info["max"])
        for cn, info in data["resources"].items()
        if info["unlimited"] or info["max"] > 0
    }

    say("Done", 1.0)
    return {
        "schematics":         sorted(set(schematics)),
        "recipes":            recipes,
        "machines":           _machines_for(recipes, data),
        "resources":          resources,       # None == unlimited (water)
        "unknown_schematics": sorted(set(unknown)),
        "parser":             parser_used,
    }


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else (
        r"C:\Users\frogy\AppData\Local\FactoryGame\Saved\SaveGames"
        r"\76561199101006188\Satisfaction 35.sav")
    r = parse_save(p)
    print(f"parser           : {r['parser']}")
    print(f"schematics       : {len(r['schematics'])}")
    print(f"unlocked recipes : {len(r['recipes'])}")
    print(f"unlocked machines: {len(r['machines'])}")
    print(f"unknown (modded) : {len(r['unknown_schematics'])} {r['unknown_schematics'][:5]}")
    print(f"machines         : {r['machines']}")


# ---------------------------------------------------------------------------
# Out-of-process parsing
# ---------------------------------------------------------------------------
#
# The vendored decoder is pure Python, so it holds the GIL for the entire 40-50
# seconds a late-game save takes. Run in a thread it starves the web server and
# every other request hangs until it finishes - which is what made imports look
# broken. A separate process sidesteps the GIL entirely.

def _parse_worker(path, queue):
    """Child-process entry point: parse and post progress back over `queue`."""
    try:
        def prog(stage, frac):
            queue.put(("progress", stage, float(frac)))
        result = parse_save(path, progress=prog)
        queue.put(("done", result))
    except Exception as exc:                          # noqa: BLE001
        queue.put(("error", f"{exc}"))


def start_parse_process(path):
    """Kick off an out-of-process parse. Returns (process, queue)."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")                     # Windows has no fork
    queue = ctx.Queue()
    proc = ctx.Process(target=_parse_worker, args=(path, queue), daemon=True)
    proc.start()
    return proc, queue
