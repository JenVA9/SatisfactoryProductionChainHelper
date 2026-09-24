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
import threading


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

# Extractors are buildings too - needed so the calculator can cost raw extraction.
EXTRACTOR_NATIVE_CLASSES = (
    "FGBuildableResourceExtractor",
    "FGBuildableWaterPump",
    "FGBuildableFrackingActivator",
)

# Map-wide resource availability (items/min) for the whole Satisfactory map.
# These match the values satisfactorytools.com sends as `resourceMax`.
# Water is deliberately unlimited - it is a plentiful resource.
RESOURCE_MAX = {
    "Desc_OreIron_C":     92100,
    "Desc_OreCopper_C":   36900,
    "Desc_Stone_C":       69900,
    "Desc_Coal_C":        42300,
    "Desc_OreGold_C":     15000,
    "Desc_LiquidOil_C":   12600,
    "Desc_RawQuartz_C":   13500,
    "Desc_Sulfur_C":      10800,
    "Desc_OreBauxite_C":  12300,
    "Desc_OreUranium_C":   2100,
    "Desc_NitrogenGas_C": 12000,
    "Desc_SAM_C":         10200,
}
UNLIMITED = float("inf")
UNLIMITED_RESOURCES = {"Desc_Water_C"}

# Recipe class paths inside a schematic's mUnlocks entry.
_RECIPE_PATH_RE = re.compile(r"\.([A-Za-z0-9_]+_C)'")


def _parse_item_list(raw: str, fluids: set = frozenset()) -> list:
    """
    Parse mIngredients / mProduct Unreal string into [{item, amount}].

    Fluids and gases are stored in the Docs JSON in litres - i.e. multiplied
    by 1000 - while solids are plain item counts. Left unscaled, a recipe like
    Pure Iron Ingot reads as "7 ore + 4000 water" instead of "7 ore + 4 water",
    which wrecks any downstream rate maths.
    """
    items   = _ITEM_RE.findall(raw)
    amounts = _AMT_RE.findall(raw)
    out = []
    for item, amt in zip(items, amounts):
        value = float(amt)
        if item in fluids:
            value /= 1000.0
        out.append({"item": item, "amount": value})
    return out


def _parse_produced_in(raw: str) -> list:
    """Parse mProducedIn into machine building classnames, excluding workbenches."""
    all_classes = _BUILD_RE.findall(raw)
    return [c for c in all_classes
            if not any(skip in c for skip in _SKIP_BUILDINGS)]


def _float_or(value, default: float) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _power_of(cls: dict) -> float:
    """
    Power draw in MW at 100% clock.

    Variable-power machines (Converter, Hadron Collider, Quantum Encoder)
    report mPowerConsumption = 0 and instead declare a min/max range; we use
    the midpoint, which is what the in-game power graph averages out to.
    """
    power = _float_or(cls.get("mPowerConsumption"), 0.0)
    if power > 0:
        return power
    lo = _float_or(cls.get("mEstimatedMininumPowerConsumption"), 0.0)
    hi = _float_or(cls.get("mEstimatedMaximumPowerConsumption"), 0.0)
    if hi > 0:
        return (lo + hi) / 2.0
    return 0.0


def _parse_unlocked_recipes(schematic: dict) -> list:
    """Recipe classNames unlocked by one FGSchematic entry."""
    out = []
    for unlock in schematic.get("mUnlocks") or []:
        if not isinstance(unlock, dict):
            continue
        # The class is BP_UnlockRecipe_C (not FGUnlockRecipe as often assumed).
        if "UnlockRecipe" not in str(unlock.get("Class", "")):
            continue
        out.extend(_RECIPE_PATH_RE.findall(unlock.get("mRecipes", "") or ""))
    return out


# ---------------------------------------------------------------------------
# Path finder
# ---------------------------------------------------------------------------

def _find_docs(hint: str = None) -> str:
    if hint and os.path.exists(hint):
        return hint

    here = os.path.dirname(os.path.abspath(__file__))
    roots = [
        # Shipped alongside the app. On a server there is no Steam install and
        # no D: drive, so without this every worker downloads the ~20MB locale
        # file from GitHub on boot - and the mirror tracks a slightly different
        # game version (200 items / 569 schematics vs 205 / 574 locally).
        os.path.join(here, "Docs"),
        os.path.join(here, "data"),
        "/opt/satisfactory/Docs",
        os.path.expanduser("~/.steam/steam/steamapps/common/Satisfactory/CommunityResources/Docs"),
        os.path.expanduser("~/.local/share/Steam/steamapps/common/Satisfactory/CommunityResources/Docs"),
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
# Several users can hit a cold server at once; without this each of them would
# parse the ~20MB locale file separately.
_cache_lock = threading.Lock()


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
    if _cache is not None:
        return _cache
    with _cache_lock:
        if _cache is not None:
            return _cache
        return _load_docs_locked(path)


def _load_docs_locked(path: str = None) -> dict:
    global _cache

    # Try local first
    try:
        resolved = _find_docs(path)
        print(f"[docs_parser] Loading local: {resolved}")
        with open(resolved, encoding="utf-16") as f:
            raw = json.load(f)
    except FileNotFoundError:
        # Fall back to online mirror
        print("[docs_parser] Local file not found, fetching online mirror...")
        import urllib.request
        req = urllib.request.Request(_ONLINE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = json.loads(r.read().decode("utf-16"))
        print("[docs_parser] Online mirror loaded.")

    # Pass 0: which item classes are fluids/gases? Needed before recipes are
    # parsed, and group order in the JSON is not guaranteed.
    fluids = set()
    for group in raw:
        if any(x in group.get("NativeClass", "") for x in ITEM_NATIVE_CLASSES):
            for cls in group.get("Classes", []):
                if str(cls.get("mForm", "")).upper() in ("RF_LIQUID", "RF_GAS"):
                    cn = cls.get("ClassName", "")
                    if cn:
                        fluids.add(cn)

    items      = {}
    recipes    = {}
    buildings  = {}
    schematics = {}
    resources  = {}

    for group in raw:
        nc      = group.get("NativeClass", "")
        classes = group.get("Classes", [])

        # Items
        if any(x in nc for x in ITEM_NATIVE_CLASSES):
            is_resource = "FGResourceDescriptor" in nc
            for cls in classes:
                cn = cls.get("ClassName", "")
                if cn:
                    items[cn] = {
                        "name":      cls.get("mDisplayName", cn),
                        "className": cn,
                        "resource":  is_resource,
                        "fluid":     cn in fluids,
                    }
                    if is_resource:
                        resources[cn] = {
                            "name":      cls.get("mDisplayName", cn),
                            "className": cn,
                            "max":       UNLIMITED if cn in UNLIMITED_RESOURCES
                                         else float(RESOURCE_MAX.get(cn, 0)),
                            "unlimited": cn in UNLIMITED_RESOURCES,
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

                ingredients = _parse_item_list(cls.get("mIngredients", ""), fluids)
                products    = _parse_item_list(cls.get("mProduct", ""), fluids)
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

        # Schematics (what each milestone / MAM / hard-drive unlock gives you)
        elif "FGSchematic" in nc:
            for cls in classes:
                cn = cls.get("ClassName", "")
                if not cn:
                    continue
                unlocked = _parse_unlocked_recipes(cls)
                # Keep schematics with no recipe unlocks too (MAM research that
                # grants inventory slots, items, etc.) so importers can tell a
                # vanilla no-op apart from a modded/unknown schematic.
                schematics[cn] = {
                    "name":      cls.get("mDisplayName", cn),
                    "className": cn,
                    "type":      cls.get("mType", ""),
                    "tier":      _float_or(cls.get("mTechTier"), 0),
                    "recipes":   unlocked,
                }

        # Extractors (miners, pumps, fracking) - buildings with a power cost
        elif any(x in nc for x in EXTRACTOR_NATIVE_CLASSES):
            for cls in classes:
                cn = cls.get("ClassName", "")
                if not cn:
                    continue
                buildings[cn] = {
                    "name":      cls.get("mDisplayName", cn),
                    "className": cn,
                    "power":     _power_of(cls),
                    "powerExponent": _float_or(cls.get("mPowerConsumptionExponent"), 1.321929),
                    "extractor": True,
                    "metadata":  {"manufacturingSpeed": 1.0},
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
                    "power":     _power_of(cls),
                    "powerExponent": _float_or(cls.get("mPowerConsumptionExponent"), 1.321929),
                    "metadata":  {"manufacturingSpeed": speed},
                }

    _cache = {
        "items":      items,
        "recipes":    recipes,
        "buildings":  buildings,
        "schematics": schematics,
        "resources":  resources,
    }
    print(f"[docs_parser] Loaded {len(items)} items, {len(recipes)} recipes, "
          f"{len(buildings)} buildings, {len(schematics)} schematics, {len(resources)} resources")
    return _cache