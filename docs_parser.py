# -*- coding: utf-8 -*-
"""
docs_parser.py

Parses the Satisfactory game's Docs/<locale>.json (UTF-16) into the
same data shape that ShareCodeResolver.py uses:

  {
    "items":     { "Desc_IronPlate_C":       { "name": "Iron Plate", ... } },
    "recipes":   { "Recipe_IronPlate_C":     { "name": ..., "time": 6.0,
                                               "ingredients": [{item, amount}],
                                               "products":    [{item, amount}],
                                               "inMachine": True, "alternate": False } },
    "buildings": { "Build_ConstructorMk1_C": { "name": "Constructor",
                                               "metadata": { "manufacturingSpeed": 1.0 } } },
  }

Usage:
    from docs_parser import load_docs
    data = load_docs(r"D:\\...\\Docs\\en-GB.json")
    # or auto-detect:
    data = load_docs()
"""

import json
import re
import os


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Extract class name: last segment after a dot, ending in _C, before ' or "
_ITEM_RE  = re.compile(r'\.([A-Za-z0-9_]+_C)[\'"]')
_AMT_RE   = re.compile(r'Amount=([0-9]+(?:\.[0-9]+)?)')
_BUILD_RE = re.compile(r'\.([A-Za-z0-9_]+_C)"')

_SKIP_BUILDINGS = {
    "WorkBench", "WorkShop", "BuildGun",
    "WorkBenchComponent", "AutomatedWorkBench", "Workshop",
}

ITEM_NATIVE_CLASSES = (
    "FGItemDescriptor", "FGResourceDescriptor",
    "FGItemDescriptorBiomass", "FGConsumableDescriptor",
    "FGEquipmentDescriptor", "FGItemDescriptorNuclearFuel",
    "FGPowerShardDescriptor", "FGItemDescriptorPowerBoosterFuel",
    "FGAmmoTypeProjectile", "FGAmmoTypeSpreadshot", "FGAmmoTypeInstantHit",
    "FGVehicleDescriptor", "FGConsumableEquipment",
)

MANUFACTURER_NATIVE_CLASSES = (
    "FGBuildableManufacturer",
    "FGBuildableManufacturerVariablePower",
)


def _parse_item_list(raw: str) -> list:
    """Parse mIngredients / mProduct Unreal string into [{item, amount}]."""
    items   = _ITEM_RE.findall(raw)
    amounts = _AMT_RE.findall(raw)
    return [{"item": item, "amount": float(amt)}
            for item, amt in zip(items, amounts)]


def _parse_produced_in(raw: str) -> list:
    """Parse mProducedIn into machine building classnames, excluding workbenches."""
    all_classes = _BUILD_RE.findall(raw)
    return [c for c in all_classes
            if not any(skip in c for skip in _SKIP_BUILDINGS)]


# ---------------------------------------------------------------------------
# Path finder
# ---------------------------------------------------------------------------

def _find_docs(hint: str = None) -> str:
    if hint and os.path.exists(hint):
        return hint

    roots = [
        r"D:\SteamLibrary\steamapps\common\Satisfactory\CommunityResources\Docs",
        r"C:\Program Files (x86)\Steam\steamapps\common\Satisfactory\CommunityResources\Docs",
        r"C:\Program Files\Steam\steamapps\common\Satisfactory\CommunityResources\Docs",
        r"D:\Steam\steamapps\common\Satisfactory\CommunityResources\Docs",
        r"E:\Steam\steamapps\common\Satisfactory\CommunityResources\Docs",
        r"E:\SteamLibrary\steamapps\common\Satisfactory\CommunityResources\Docs",
    ]
    for root in roots:
        for locale in ("en-US.json", "en-GB.json", "en-AU.json"):
            p = os.path.join(root, locale)
            if os.path.exists(p):
                return p

    raise FileNotFoundError(
        "Could not find Satisfactory Docs JSON automatically. "
        "Pass the path explicitly: load_docs(path=r'...')"
    )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

_cache: dict | None = None


# Online fallback - same file format, kept in sync with game updates by the community
_ONLINE_URL = "https://raw.githubusercontent.com/aringadre76/satisfactory-api/main/Docs/en-US.json"


def load_docs(path: str = None) -> dict:
    """
    Parse a Satisfactory Docs locale JSON and return a data dict
    compatible with ShareCodeResolver.

    Priority:
      1. Local game install (auto-detected or explicit path)
      2. Online community mirror on GitHub (no Render cold-start)

    Returns cached result on repeated calls.
    """
    global _cache
    if _cache is not None:
        return _cache

    # Try local first
    try:
        resolved = _find_docs(path)
        print(f"[docs_parser] Loading local: {resolved}")
        with open(resolved, encoding="utf-16") as f:
            raw = json.load(f)
    except FileNotFoundError:
        # Fall back to online mirror
        print(f"[docs_parser] Local file not found, fetching online mirror...")
        import urllib.request
        req = urllib.request.Request(_ONLINE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = json.loads(r.read().decode("utf-16"))
        print(f"[docs_parser] Online mirror loaded.")

    items     = {}
    recipes   = {}
    buildings = {}

    for group in raw:
        nc      = group.get("NativeClass", "")
        classes = group.get("Classes", [])

        # Items
        if any(x in nc for x in ITEM_NATIVE_CLASSES):
            for cls in classes:
                cn = cls.get("ClassName", "")
                if cn:
                    items[cn] = {
                        "name":      cls.get("mDisplayName", cn),
                        "className": cn,
                    }

        # Recipes
        elif "FGRecipe" in nc and "Customization" not in nc:
            for cls in classes:
                cn = cls.get("ClassName", "")
                if not cn:
                    continue

                produced_in = _parse_produced_in(cls.get("mProducedIn", ""))
                if not produced_in:
                    continue   # hand-crafted / build-gun only

                ingredients = _parse_item_list(cls.get("mIngredients", ""))
                products    = _parse_item_list(cls.get("mProduct", ""))
                if not products:
                    continue

                try:
                    time = float(cls.get("mManufactoringDuration", 0))
                except (ValueError, TypeError):
                    continue
                if time <= 0:
                    continue

                name = cls.get("mDisplayName", cn)
                recipes[cn] = {
                    "name":        name,
                    "className":   cn,
                    "alternate":   "Alternate" in name,
                    "time":        time,
                    "inMachine":   True,
                    "ingredients": ingredients,
                    "products":    products,
                    "producedIn":  produced_in,
                }

        # Manufacturer buildings
        elif any(x in nc for x in MANUFACTURER_NATIVE_CLASSES):
            for cls in classes:
                cn = cls.get("ClassName", "")
                if not cn:
                    continue
                try:
                    speed = float(cls.get("mManufacturingSpeed", 1.0))
                except (ValueError, TypeError):
                    speed = 1.0
                buildings[cn] = {
                    "name":      cls.get("mDisplayName", cn),
                    "className": cn,
                    "metadata":  {"manufacturingSpeed": speed},
                }

    _cache = {"items": items, "recipes": recipes, "buildings": buildings}
    print(f"[docs_parser] Loaded {len(items)} items, {len(recipes)} recipes, {len(buildings)} buildings")
    return _cache