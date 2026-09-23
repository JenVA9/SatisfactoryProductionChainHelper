# -*- coding: utf-8 -*-
"""
calculator.py

Production calculator for Satisfactory.

Builds a linear model over the unlocked recipe set and solves it in one of
four modes. The result is emitted in exactly the shape ShareCodeResolver's
solver returns, so it can reuse the same graph builder and layering code -
which means the existing front-end renders a calculated chain identically to
an imported one.

Public API
----------
build_model(world=None, blocked_recipes=(), blocked_machines=())
calculate(targets, mode="least_power", depth="quick", world=None, ...)

targets: list, in priority order (top = highest priority):
    [{"item": "Desc_IronPlate_C", "type": "exact",    "amount": 30},
     {"item": "Desc_IronRod_C",   "type": "maximize"}]

modes: "least_power" | "least_machines" | "least_steps" | "easiest" | "maximise"
depth: "quick" (LP relaxation / capped MILP) | "absolute" (exact MILP)
"""

import copy
import math
import os
import threading
from concurrent import futures
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

from docs_parser import load_docs

EPS = 1e-7
BIG_M = 1e5          # upper bound on machine count for indicator constraints

# Absolute depth is meant to be trustworthy, not merely "kinda working", so it
# gets real headroom to prove optimality before giving up.
ABSOLUTE_TIME_LIMIT = 30

# What "easiest" will pay, in recipe-steps, to avoid one leftover fluid.
FLUID_BYPRODUCT_COST = 3.0

# Where several plans score the same on a mode's own objective, prefer the one
# with fewer buildings - that is always worth having. Small enough to only ever
# break ties, never to override the mode's actual goal.
MACHINE_TIEBREAK = 1e-4

# "Easiest" is the beginner middle ground between fewest machines and fewest
# steps, leaning towards fewer steps: ~20 machines are worth one extra step.
EASIEST_MACHINE_WEIGHT = 0.05

# How hard stage 2 pulls towards the requested amount. This has to dominate the
# machine cost: the amount is a ceiling, but any shortfall should come from the
# nice-number lattice not quite reaching it, never from the solver deciding that
# building nothing is cheaper. (At 0.4 the all-zero plan won outright.)
OUTPUT_PULL = 8.0

# Above this many distinct recipes the nice-number pass is skipped - it is one
# binary per (recipe x fraction), so it grows quickly and stops being worth the
# wait. Chains this wide are past "beginner" territory anyway.
EASY_NUMBERS_MAX_RECIPES = 90

# What one extra resource type must save, as a share of the mode's objective,
# before "soft" will take it.
SOFT_RESOURCE_WEIGHT = 0.35

# Cost of letting one stage sit on an awkward clock so the chain can balance
# exactly. Dear enough that most stages stay tidy, cheap enough that it is
# always preferred over overproducing.
FREE_FRACTION_COST = 0.6

# Granularity of that escape hatch: 1e-4 of a machine is 0.01% of clock, so any
# fraction it picks still prints as at most two decimal places - which is the
# limit of what is reasonable to dial in by hand.
FREE_STEP = 1e-4

# How much of the requested amount a zero-overproduction plan must still make
# before it is preferred over a plan that hits the number but leaves spares.
# Deliberately low: "Easiest" exists to produce something clean and buildable,
# and the requested amount is a ceiling. Overproducing to hit the number exactly
# is the thing this mode is supposed to avoid, so a short but spotless plan wins
# over a full one that leaves piles of spare Circuit Boards. The shortfall is
# reported so it is never a silent surprise.
STRICT_MIN_SCORE = 0.2

# Surplus below this (items/min) is rounding residue, not a byproduct worth
# plumbing away. It also has to be non-zero: machine counts are quantised to
# 1e-4, so demanding an *exactly* zero balance made the no-overproduction
# program infeasible at every output except zero, and it silently lost to the
# overproducing plan every time.
STRICT_SURPLUS_TOL = 0.05

# When stage 1's own counts are undialable, take a nice-numbered plan that makes
# at least this share of the target in preference to them.
NICE_MIN_SCORE = 0.2

# Floor for a MAX / LMAX target, items per minute.
#
# Leximin maximises the *worst* target, so one item that cannot reach the common
# level pins that level at zero - and a zero-rate target was then dropped from
# the demand entirely, which is why asking for rockets alongside cheaper items
# produced no rockets at all. Asking to maximise something is still asking for
# it, so every maximised target is pinned at least this high.
MIN_MAXIMISE_RATE = 0.01

# A machine count below this is solver noise, not a decision. HiGHS will
# happily leave a recipe running at 0.0000016 of a machine to soak up a
# rounding residue; that became a real card, edge and layer in the chain,
# advertising 0.0003/min of something nothing uses. Below 1% clock nothing is
# buildable anyway, so a count this small can only be an artefact.
NOISE_MACHINE_COUNT = 1e-3

# --- LMAX fairness ---------------------------------------------------------
# How evenly linked targets share what is left once everyone is guaranteed the
# equal-split level. This is the alpha in alpha-fair allocation:
#   0   - maximise the raw total; whichever target is cheapest takes everything
#   1   - proportional (Nash) fairness; a target's marginal value is 1/its rate
#   >1  - progressively flatter, approaching leximin as it grows
LMAX_FAIRNESS = 1.0

# How linked targets share whatever is spare once everyone is guaranteed the
# equal-split level:
#   "headroom"     - raise every target by the same fraction of its own
#                    remaining room, so the one that could have gone highest
#                    ends up highest and they all move together
#   "proportional" - alpha-fair by cost; the cheapest target takes the spare
#                    capacity and the rest stay on the floor
LMAX_MODE = "headroom"

# --- letting MAX / LMAX output flex ----------------------------------------
# How far a settled MAX/LMAX level may be walked back, smallest step first, and
# how much the mode's own metric has to improve before giving up that output is
# considered worth it. Exact targets are never touched.
MAX_REDUCE_STEPS = (0.02, 0.05, 0.10)
MAX_REDUCE_MIN_GAIN = 0.05

# The fairness curve is concave, so it is fed to the LP as a piecewise-linear
# approximation: geometric segments between each target's guaranteed floor and
# the most it could ever reach on its own.
LMAX_SEGMENTS = 28

# Extraction is not free: pulling a resource costs extractor buildings and
# power. Without this the solver treats unlimited water as costless and will
# happily pump tens of thousands a minute to save one machine downstream.
# (rate per extractor in items/min, power draw in MW). Solid ores assume a
# Mk2 miner on a normal node, which is the sane default when we deliberately
# do not model node purity.
_EXTRACTORS = {
    "Desc_Water_C":       (120.0,  20.0),   # Water Extractor
    "Desc_LiquidOil_C":   (120.0,  40.0),   # Oil Extractor
    "Desc_NitrogenGas_C": (120.0, 150.0),   # Resource Well Pressuriser
}
_DEFAULT_EXTRACTOR = (120.0, 12.0)          # Miner Mk2, normal node


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Model:
    """Linear model: variables = machine counts per recipe, then extraction."""

    def __init__(self, data, recipe_ids, machine_of, net, items,
                 raw_items, raw_limits, power):
        self.data = data
        self.recipe_ids = recipe_ids            # list[str]
        self.machine_of = machine_of            # recipe -> building class
        self.net = net                          # ndarray (n_items, n_recipes)
        self.items = items                      # list[str]
        self.item_ix = {it: i for i, it in enumerate(items)}
        self.raw_items = raw_items              # list[str]
        self.raw_ix = {it: i for i, it in enumerate(raw_items)}
        self.raw_limits = raw_limits            # ndarray, inf allowed
        self.power = power                      # ndarray (n_recipes,) MW
        rates = np.array([_EXTRACTORS.get(it, _DEFAULT_EXTRACTOR)[0]
                          for it in raw_items]) if raw_items else np.zeros(0)
        mw = np.array([_EXTRACTORS.get(it, _DEFAULT_EXTRACTOR)[1]
                       for it in raw_items]) if raw_items else np.zeros(0)
        self.raw_power = mw / rates if len(rates) else np.zeros(0)   # MW per item/min
        self.raw_machines = 1.0 / rates if len(rates) else np.zeros(0)
        # Fluids cannot be sunk and need pipes, sinks or a whole disposal loop,
        # so a leftover fluid is far more annoying than a leftover solid.
        self.is_fluid = np.array(
            [bool(data["items"].get(it, {}).get("fluid")) for it in items])

    @property
    def n_r(self):
        return len(self.recipe_ids)

    @property
    def n_raw(self):
        return len(self.raw_items)

    @property
    def n_var(self):
        return self.n_r + self.n_raw


def _rate(entry, time, speed):
    return entry["amount"] * (60.0 / time) * speed


def build_model(world=None, blocked_recipes=(), blocked_machines=(),
                resource_limits=None) -> Model:
    """
    Build the model. `world` is the dict from save_parser.parse_save(), or None
    for the no-world default (everything unlocked, map-wide resource limits).

    `resource_limits` overrides how much of a raw resource may be used:
        {"Desc_OreIron_C": 120}   cap iron ore at 120/min
        {"Desc_Coal_C": 0}        do not use coal at all
        {"Desc_Water_C": None}    unlimited
    Anything not mentioned keeps its default.
    """
    data = load_docs()
    recipes, buildings = data["recipes"], data["buildings"]

    allowed_recipes = set(world["recipes"]) if world else set(recipes)
    allowed_machines = set(world["machines"]) if world else None
    blocked_recipes = set(blocked_recipes)
    blocked_machines = set(blocked_machines)

    recipe_ids, machine_of = [], {}
    for rc, r in recipes.items():
        if rc not in allowed_recipes or rc in blocked_recipes:
            continue
        # Pick a machine this recipe can run in that is actually available.
        machine = None
        for mch in r.get("producedIn", []):
            if mch in blocked_machines:
                continue
            if allowed_machines is not None and mch not in allowed_machines:
                continue
            if mch in buildings:
                machine = mch
                break
        if machine is None:
            continue
        recipe_ids.append(rc)
        machine_of[rc] = machine

    # Item universe
    item_set = set()
    for rc in recipe_ids:
        r = recipes[rc]
        for e in r["ingredients"] + r["products"]:
            item_set.add(e["item"])
    items = sorted(item_set)
    item_ix = {it: i for i, it in enumerate(items)}

    # Net production matrix, per machine per minute
    net = np.zeros((len(items), len(recipe_ids)))
    power = np.zeros(len(recipe_ids))
    for j, rc in enumerate(recipe_ids):
        r = recipes[rc]
        b = buildings[machine_of[rc]]
        speed = b.get("metadata", {}).get("manufacturingSpeed", 1.0)
        t = r["time"]
        for e in r["ingredients"]:
            net[item_ix[e["item"]], j] -= _rate(e, t, speed)
        for e in r["products"]:
            net[item_ix[e["item"]], j] += _rate(e, t, speed)
        power[j] = b.get("power", 0.0)

    # Raw resources: only those that actually appear in this model.
    limits = world["resources"] if world else {
        cn: (None if info["unlimited"] else info["max"])
        for cn, info in data["resources"].items()
        if info["unlimited"] or info["max"] > 0
    }
    if resource_limits:
        for cn, cap in resource_limits.items():
            if cn in limits:
                limits[cn] = None if cap is None else max(0.0, float(cap))

    raw_items = [it for it in items if it in limits]
    raw_limits = np.array(
        [np.inf if limits[it] is None else float(limits[it]) for it in raw_items]
    )

    return Model(data, recipe_ids, machine_of, net, items,
                 raw_items, raw_limits, power)


# ---------------------------------------------------------------------------
# Constraint assembly
# ---------------------------------------------------------------------------

def _balance_rows(m: Model):
    """
    Coefficient block enforcing  net.x + extraction >= demand  per item.
    Demand is supplied by the caller.
    """
    block = np.zeros((len(m.items), m.n_var))
    block[:, :m.n_r] = m.net
    for k, it in enumerate(m.raw_items):
        block[m.item_ix[it], m.n_r + k] = 1.0
    return block


def _bounds(m: Model, extra=0):
    lo = np.zeros(m.n_var + extra)
    hi = np.full(m.n_var + extra, np.inf)
    hi[m.n_r:m.n_r + m.n_raw] = m.raw_limits
    return lo, hi


def _demand_vector(m: Model, fixed: dict):
    d = np.zeros(len(m.items))
    for it, amt in fixed.items():
        if it in m.item_ix:
            d[m.item_ix[it]] = amt
    return d


# ---------------------------------------------------------------------------
# Satisfy exact targets, then equalise + prioritise the maximised ones
# ---------------------------------------------------------------------------

def _solo_ceiling(m: Model, floors: dict, item: str, bnds, block):
    """The most `item` could reach if only the exact targets had to be met."""
    n = m.n_var
    dem = _demand_vector(m, floors)
    # The row for the item being measured carries no demand of its own: the
    # constraint is net - t >= 0, so t IS the rate. Leaving a floor in there
    # instead gives net - t >= floor, i.e. t one whole floor short of the truth
    # - invisible at 100k/min, but a 0.3% haircut on a target that runs at 3.
    dem[m.item_ix[item]] = 0.0
    A = np.hstack([block, np.zeros((block.shape[0], 1))])
    A[m.item_ix[item], n] = -1.0
    c = np.zeros(n + 1)
    c[n] = -1.0
    r = linprog(c, A_ub=-A, b_ub=-dem, bounds=bnds, method="highs")
    return max(0.0, float(r.x[n])) if r.success else 0.0


def _equal_split_level(m: Model, floors: dict, items: list, bnds, block):
    """The common level every one of `items` can reach together - the floor."""
    n = m.n_var
    dem = _demand_vector(m, floors)
    for it in items:                       # same reasoning as _solo_ceiling
        dem[m.item_ix[it]] = 0.0
    A = np.hstack([block, np.zeros((block.shape[0], 1))])
    for it in items:
        A[m.item_ix[it], n] = -1.0
    c = np.zeros(n + 1)
    c[n] = -1.0
    r = linprog(c, A_ub=-A, b_ub=-dem, bounds=bnds, method="highs")
    return max(0.0, float(r.x[n])) if r.success else 0.0


def _headroom_levels(m: Model, exact: dict, items: list, floor: float,
                     ceilings: dict):
    """
    Raise every target by the same fraction of its own remaining room, then
    lock whatever has run out and carry on with the rest.

    A single pass is not enough. One shared fraction couples every target, so a
    target whose ceiling is twenty times larger needs a huge absolute rise for
    any fraction at all - the shared resources run out immediately, the
    fraction pins near zero, and everything lands flat on the floor. That cost
    40% of the output on a real oil chain.

    Iterating fixes it: targets with comparable room rise together and stay
    graded by room (the 3 / 3.25 / 3.5 shape), and once the small ones top out
    the roomy ones keep climbing on their own instead of being held back.
    """
    live = [it for it in items if it in m.item_ix]
    if not live:
        return {}

    base = max(floor, MIN_MAXIMISE_RATE)
    current = {it: base for it in live}
    remaining = list(live)
    n = m.n_var
    block = _balance_rows(m)
    bnds_plain = list(zip(*_bounds(m, extra=1)))

    for _round in range(len(live) + 2):
        if not remaining:
            break

        # Room left for each target, with everything else held where it is.
        room = {}
        for it in remaining:
            floors = dict(exact)
            for o in live:
                if o != it:
                    floors[o] = current[o]
            ceil = _solo_ceiling(m, floors, it, bnds_plain, block)
            room[it] = max(0.0, ceil - current[it])

        movable = [it for it in remaining if room[it] > max(1e-9, current[it] * 1e-9)]
        if not movable:
            break

        lo, hi = _bounds(m, extra=1)
        hi[n] = 1.0
        A = np.hstack([block, np.zeros((block.shape[0], 1))])
        for it in movable:
            A[m.item_ix[it], n] = -room[it]
        c = np.zeros(n + 1)
        c[n] = -1.0

        floors = dict(exact)
        for it in live:
            floors[it] = current[it]
        dem = _demand_vector(m, floors)

        res = linprog(c, A_ub=-A, b_ub=-dem, bounds=list(zip(lo, hi)),
                      method="highs")
        if not res.success:
            break
        t = min(1.0, max(0.0, float(res.x[n])))
        if t <= 1e-9:
            break

        for it in movable:
            current[it] = current[it] + room[it] * t

        # Whatever has reached its own ceiling drops out at the top of the next
        # round, where the room is recomputed anyway - re-measuring every
        # ceiling again here just doubled the solve count for the same answer.
        remaining = list(movable)

    return current


def _fair_levels(m: Model, exact: dict, items: list, floor: float, ceilings: dict):
    """
    Share capacity above `floor` by alpha-fair allocation.

    One extra variable per target per segment. Segment k of target i may only be
    used once the segments below it are full, which the LP arranges for free
    because the weights decrease - so the piecewise sum behaves like a concave
    utility without needing any integer variables.
    """
    n = m.n_var
    block = _balance_rows(m)
    live = [it for it in items if it in m.item_ix]

    # Segment layout per target: geometric from its floor to its own ceiling.
    segs = {}
    for it in live:
        base = max(floor, MIN_MAXIMISE_RATE)
        top = max(ceilings.get(it, base), base * (1.0 + 1e-9))
        if top <= base * (1.0 + 1e-9):
            segs[it] = []                       # no headroom; pinned at floor
            continue
        ratio = (top / base) ** (1.0 / LMAX_SEGMENTS)
        edges = [base * (ratio ** k) for k in range(LMAX_SEGMENTS + 1)]
        segs[it] = [(edges[k], edges[k + 1] - edges[k])
                    for k in range(LMAX_SEGMENTS)]

    n_seg = sum(len(v) for v in segs.values())
    if n_seg == 0:
        return {it: floor for it in live}

    total = n + n_seg
    lo = np.zeros(total)
    hi = np.full(total, np.inf)
    hi[m.n_r:m.n_r + m.n_raw] = m.raw_limits

    A = np.zeros((len(m.items), total))
    A[:, :n] = block
    c = np.zeros(total)

    col = n
    for it in live:
        row = m.item_ix[it]
        for left, width in segs[it]:
            A[row, col] = -1.0                  # net_i - sum(segments) >= floor
            hi[col] = width
            # Marginal value of output at `left`. alpha=1 gives 1/left, i.e.
            # proportional fairness; the scale cancels, so targets measured in
            # thousands and targets measured in ones compete on equal terms.
            c[col] = -(left ** -LMAX_FAIRNESS)
            col += 1

    floors = dict(exact)
    for it in live:
        floors[it] = max(floor, MIN_MAXIMISE_RATE)
    dem = _demand_vector(m, floors)

    res = linprog(c, A_ub=-A, b_ub=-dem, bounds=list(zip(lo, hi)),
                  method="highs")
    if not res.success:
        return None

    out = {}
    col = n
    for it in live:
        got = max(floor, MIN_MAXIMISE_RATE)
        for _left, _w in segs[it]:
            got += float(res.x[col])
            col += 1
        out[it] = got
    return out


def _solve_levels_linked(m: Model, exact: dict, linked: list):
    """
    Unbiased ("linked") maximisation - leximin.

    Ordinary MAX raises everything to a common level and then walks the list in
    priority order, so whatever sits at the top wins the contested resources.
    LINKED MAX has no such bias: it raises every target together until one of
    them can go no further, locks that one where it stopped, and carries on
    raising the rest. Repeating that shares the resources out evenly instead of
    front-loading the list.
    """
    live = [it for it in linked if it in m.item_ix]
    if not live:
        return {}

    block = _balance_rows(m)
    lo, hi = _bounds(m, extra=1)
    bnds = list(zip(lo, hi))
    n = m.n_var

    # Everyone is guaranteed the level an equal split would have reached, then
    # fairness decides who climbs above it. Without this second step a target
    # with its own spare resource just sat at the common level, because lifting
    # it did nothing for the worst-off target and leximin only cares about that.
    floors = dict(exact)
    for it in live:
        floors.setdefault(it, MIN_MAXIMISE_RATE)
    guaranteed = _equal_split_level(m, floors, live, bnds, block)
    if guaranteed > 0.0:
        ceilings = {it: _solo_ceiling(m, floors, it, bnds, block) for it in live}
        shareout = (_headroom_levels if LMAX_MODE == "headroom" else _fair_levels)
        got = shareout(m, exact, live, guaranteed, ceilings)
        if got:
            return got

    # Fallback: the original leximin staircase.
    settled: dict = {}
    remaining = list(live)
    guard = 0
    while remaining and guard <= len(live):
        guard += 1
        floors = dict(exact)
        for it in live:
            floors.setdefault(it, MIN_MAXIMISE_RATE)
        dem = _demand_vector(m, floors)
        for it, v in settled.items():
            dem[m.item_ix[it]] = max(v, MIN_MAXIMISE_RATE)

        A = np.hstack([block, np.zeros((block.shape[0], 1))])
        for it in remaining:
            A[m.item_ix[it], n] = -1.0             # net_i - t >= 0
        c = np.zeros(n + 1)
        c[n] = -1.0
        res = linprog(c, A_ub=-A, b_ub=-dem, bounds=bnds, method="highs")
        if not res.success:
            break
        level = max(0.0, float(res.x[n]))

        # Which of the remaining cannot go any higher than this common level?
        # Those are the binding ones; lock them and free the rest to rise.
        stuck = []
        for it in remaining:
            d2 = dem.copy()
            for other in remaining:
                if other is not it:
                    d2[m.item_ix[other]] = level
            A2 = np.hstack([block, np.zeros((block.shape[0], 1))])
            A2[m.item_ix[it], n] = -1.0
            c2 = np.zeros(n + 1)
            c2[n] = -1.0
            r2 = linprog(c2, A_ub=-A2, b_ub=-d2, bounds=bnds, method="highs")
            best_alone = max(0.0, float(r2.x[n])) if r2.success else level
            if best_alone <= level + max(1e-6, level * 1e-6):
                stuck.append(it)

        if not stuck:                              # nothing binding - all done
            stuck = list(remaining)
        for it in stuck:
            settled[it] = level
            remaining.remove(it)

    for it in remaining:                           # safety net
        settled.setdefault(it, 0.0)
    return settled


def _solve_levels(m: Model, exact: dict, maximise: list):
    """
    Returns {item: achieved_rate} for the maximised items.

    Exact targets are hard constraints. Maximised items are first raised to a
    common level, then - in priority order - each is pushed as high as it can
    go without dropping any already-settled item. That gives equal output
    where possible, and favours the higher-priority item where it is not.
    """
    if not maximise:
        return {}

    block = _balance_rows(m)
    floors = dict(exact)
    for it in maximise:
        floors.setdefault(it, MIN_MAXIMISE_RATE)
    dem = _demand_vector(m, floors)
    n = m.n_var                                  # layout: [x..., e..., t]

    A = np.hstack([block, np.zeros((block.shape[0], 1))])
    for it in maximise:
        if it in m.item_ix:
            A[m.item_ix[it], n] = -1.0           # net - t >= 0
    lo, hi = _bounds(m, extra=1)
    bnds = list(zip(lo, hi))

    c = np.zeros(n + 1)
    c[n] = -1.0                                  # maximise t
    res = linprog(c, A_ub=-A, b_ub=-dem, bounds=bnds, method="highs")
    if not res.success:
        raise ValueError("No feasible production plan for these targets. "
                         "A resource you switched off or capped may be needed, "
                         "or the recipe is not unlocked in this world.")
    level = max(0.0, float(res.x[n]))

    # Lexicographic pass: earlier items win contested resources.
    settled = {}
    for idx, it in enumerate(maximise):
        A2 = np.hstack([block, np.zeros((block.shape[0], 1))])
        d2 = dem.copy()
        for o, v in settled.items():
            if o in m.item_ix:
                d2[m.item_ix[o]] = v
        for o in maximise[idx + 1:]:
            if o in m.item_ix:
                d2[m.item_ix[o]] = level         # others held at common level
        if it in m.item_ix:
            A2[m.item_ix[it], n] = -1.0
        c2 = np.zeros(n + 1)
        c2[n] = -1.0
        r2 = linprog(c2, A_ub=-A2, b_ub=-d2, bounds=bnds, method="highs")
        settled[it] = max(level, float(r2.x[n])) if r2.success else level

    return settled


# ---------------------------------------------------------------------------
# Optimise the chosen mode at the settled output levels
# ---------------------------------------------------------------------------

def _nice(v: float) -> float:
    """Round down to a 'nice', easily-underclocked number."""
    if v <= 0:
        return 0.0
    for step in (100, 60, 50, 30, 25, 20, 15, 12, 10, 6, 5, 4, 3, 2, 1):
        if v >= step:
            return math.floor(v / step) * step
    return math.floor(v * 2) / 2


def _trim(m: Model, dem, x):
    """
    A MILP fixes *which* recipes to run but has no reason to mine sparingly -
    extraction is nearly free, so it parks every resource at its cap and dumps
    the surplus as phantom byproducts.

    With the machine vector held exactly as the solver chose it, the minimum
    extraction is not a search at all: each raw resource only needs to cover
    whatever the recipes do not already supply. Solving it directly (rather
    than re-running an LP) keeps the solver's integer machine counts intact -
    re-optimising here used to relax them back to fractions like 0.06.
    """
    shortfall = dem - (m.net @ x)
    e = np.zeros(m.n_raw)
    for k, it in enumerate(m.raw_items):
        e[k] = max(0.0, float(shortfall[m.item_ix[it]]))
    return x, np.minimum(e, m.raw_limits)


def _surplus_cost(m: Model, solid=0.002, fluid=0.02):
    """
    Linear penalty on leftover output.

    Net production of an item is (block . vars); the balance constraint already
    forces it to at least the demand, so pushing it *down* squeezes out anything
    nobody asked for. Weights are per item/min and sized so that clearing a
    ~500/min solid byproduct, or a ~50/min fluid one, is worth about one extra
    recipe step - fluids cost ten times as much because you cannot just sink them.
    """
    w = np.where(m.is_fluid, fluid, solid)
    return w @ _balance_rows(m)


def _mode_cost_xe(m: Model, mode: str):
    """
    The mode's objective expressed over [machines, extraction].

    "maximise" deliberately has no build cost at all - output is already fixed
    by the level solve, and this mode is defined as not caring how expensive
    the factory is to run.
    """
    if mode == "least_power":
        return np.concatenate([m.power, m.raw_power])
    if mode in ("least_machines", "least_steps", "easiest"):
        return np.concatenate([np.ones(m.n_r), m.raw_machines])
    return np.zeros(m.n_var)                     # maximise


def _countable_raw(m: Model):
    """
    Raw resources that count as a distinct "node type".

    Water is deliberately excluded: it is plentiful, and the user asked for it
    to be free unless they switch it off (which sets its limit to 0, removing
    it here anyway). Anything with an infinite allowance is treated the same.
    """
    return [k for k in range(m.n_raw)
            if np.isfinite(m.raw_limits[k]) and m.raw_limits[k] > 0]


def _restrict(m: Model, keep_mask) -> Model:
    """A copy of the model with unchosen resources switched off."""
    clone = copy.copy(m)
    clone.raw_limits = np.where(keep_mask, m.raw_limits, 0.0)
    return clone


def choose_resources(m: Model, demand: dict, mode: str, strength: str,
                     maximise_items=()):
    """
    Pick the smallest sensible set of raw resources that can still make the
    targets, then hand back a mask so the real solve runs with those at their
    *full* limits and the rest switched off.

    strength:
        "hard" - fewest node types outright; output then breaks ties between
                 sets of the same size, so you get the *best* two resources
                 rather than any two that happen to work.
        "soft" - each extra node type is charged a slice of the objective, so
                 one is only added when it genuinely pays for itself.

    `maximise_items` matters a great deal. Those targets have no fixed amount,
    so a set chosen purely on "can it make the item at all" will happily pick
    one resource that makes a trickle over two that make ten times as much.
    The level variable below makes the choice output-aware.
    """
    countable = _countable_raw(m)
    keep = np.ones(m.n_raw, dtype=bool)
    if strength not in ("soft", "hard") or not countable:
        return keep

    block = _balance_rows(m)
    dem = _demand_vector(m, demand)
    lo, hi = _bounds(m)
    base = _mode_cost_xe(m, mode)

    # Scale the mode objective to ~1 so the per-resource weight means the same
    # thing whether we are counting megawatts or machines.
    obj0 = 0.0
    if np.any(base):
        ref = linprog(base, A_ub=-block, b_ub=-dem,
                      bounds=list(zip(lo, hi)), method="highs")
        if ref.success:
            obj0 = abs(float(np.dot(base, ref.x)))
    norm = (base / obj0) if obj0 > 1e-9 else np.zeros_like(base)

    nz, n = len(countable), m.n_var
    live = [it for it in maximise_items if it in m.item_ix]
    width = n + nz + (1 if live else 0)           # [x, e, z, t]
    t_ix = n + nz

    A = np.hstack([block, np.zeros((block.shape[0], width - n))])
    for it in live:                               # net_i - t >= 0
        A[m.item_ix[it], t_ix] = -1.0
    link = np.zeros((nz, width))                  # e_k - limit_k * z_j <= 0
    for j, k in enumerate(countable):
        link[j, m.n_r + k] = 1.0
        link[j, n + j] = -float(m.raw_limits[k])
    cons = [LinearConstraint(A, lb=dem, ub=np.inf),
            LinearConstraint(link, lb=-np.inf, ub=0.0)]

    # hard: node count dominates outright, everything else only breaks ties.
    # soft: one extra node type is worth ~12% of the baseline objective.
    # Soft used to sit at 0.12, which let it take an extra resource to shave a
    # single fractional machine off an estimate that did not even match the
    # final whole-building count. An extra node type now has to earn its keep.
    z_weight = 1.0 if strength == "hard" else SOFT_RESOURCE_WEIGHT
    mode_weight = 1e-3 if strength == "hard" else 1.0
    c = np.zeros(width)
    c[:n] = norm * mode_weight
    c[n:n + nz] = z_weight

    if live:
        # Reward output. Scaled against the per-resource weight so that under
        # "soft" a set that makes materially more wins, while under "hard" it
        # can only separate sets that already tie on count.
        ref_t = _max_common_level(m, block, dem, lo, hi, live)
        if ref_t > EPS:
            pull = (z_weight * 0.9) if strength == "hard" else (z_weight * 8.0)
            c[t_ix] = -pull / ref_t

    lo2 = np.concatenate([lo, np.zeros(nz)] + ([np.zeros(1)] if live else []))
    hi2 = np.concatenate([hi, np.ones(nz)] + ([np.full(1, np.inf)] if live else []))
    integ = np.concatenate([np.zeros(n), np.ones(nz)] +
                           ([np.zeros(1)] if live else []))

    r = milp(c=c, constraints=cons, integrality=integ,
             bounds=Bounds(lo2, hi2),
             options={"time_limit": 15,
                      "mip_rel_gap": 0.0 if strength == "hard" else 0.02})
    if not _usable(r):
        return keep                               # fall back to using everything

    for j, k in enumerate(countable):
        keep[k] = r.x[n + j] > 0.5
    return keep


def _max_common_level(m, block, dem, lo, hi, items):
    """Best common output level across `items` with every resource available."""
    n = m.n_var
    A = np.hstack([block, np.zeros((block.shape[0], 1))])
    for it in items:
        A[m.item_ix[it], n] = -1.0
    c = np.zeros(n + 1)
    c[n] = -1.0
    res = linprog(c, A_ub=-A, b_ub=-dem,
                  bounds=list(zip(np.concatenate([lo, [0.0]]),
                                  np.concatenate([hi, [np.inf]]))),
                  method="highs")
    return max(0.0, float(res.x[n])) if res.success else 0.0


def _usable(r):
    """
    A MILP that hits its time limit reports success=False, but it still carries
    the best plan it found. Throwing that away and falling back to a plain LP
    was losing good answers on deep chains - so accept any feasible incumbent
    and just flag the result as unproven.
    """
    return r is not None and getattr(r, "x", None) is not None


# Clock settings a beginner can actually dial in: whole machines first, then
# simple halves/quarters/thirds. Ordered best-first; the cost column is what
# "easiest" pays to use one, so it reaches for a tidy number before an odd one.
def clock_is_dialable(clock_pct):
    """
    Can a person actually set this clock?

    Allowed: anything terminating within two decimals (60.25%), and recurring
    values whose repeat is a single digit after at most two fixed decimals
    (33.33..%, 20.8333..%, 11.111..%). Those are exactly the values whose
    percentage times 900 is a whole number.

    Rejected: longer repeating blocks such as 78.148148..%, which is 211/270 -
    a ratio that falls out of the recipe maths and cannot be dialled in.
    """
    if clock_pct < 1e-9:
        return True
    return abs(clock_pct * 900 - round(clock_pct * 900)) < 1e-4


def nice_fraction_stats(x):
    """(dialable machine groups, total) for a machine-count vector."""
    total = ok = 0
    for v in x:
        if v <= EPS:
            continue
        total += 1
        if clock_is_dialable((v - math.floor(v + 1e-9)) * 100.0):
            ok += 1
    return ok, total


NICE_FRACTIONS = [
    (0.0,       0.00),   # 100% - whole machines
    (0.5,       0.05),   # 50%
    (0.25,      0.08),   # 25%
    (0.75,      0.08),   # 75%
    (1.0 / 3.0, 0.12),   # 33.33% recurring
    (2.0 / 3.0, 0.12),   # 66.67% recurring
    (0.2,       0.15),   # 20%
    (0.4,       0.15),   # 40%
    (0.6,       0.15),   # 60%
    (0.8,       0.15),   # 80%
    (0.1,       0.20),   # 10%
    (0.9,       0.20),   # 90%
    (1.0 / 6.0, 0.22),   # 16.67% recurring
    (5.0 / 6.0, 0.22),   # 83.33% recurring
    (1.0 / 12.0, 0.26),  # 8.33% recurring
    (5.0 / 12.0, 0.26),  # 41.67% recurring
    (7.0 / 12.0, 0.26),  # 58.33% recurring
    (11.0 / 12.0, 0.26), # 91.67% recurring
    (0.125,     0.28),   # 12.5%
    (0.375,     0.28),   # 37.5%
    (0.625,     0.28),   # 62.5%
    (0.875,     0.28),   # 87.5%
]


def _easy_numbers(m: Model, demand: dict, xs, depth):
    """
    Second pass for "easiest": rebuild the plan on machine counts a human can
    actually set.

    The first pass picks *which* recipes to use, but leaves counts like
    5.267489712 - a last machine at 26.748971%, which is neither a clean
    recurring nor two decimal places, so nobody can dial it in. Here each
    chosen recipe is re-expressed as `whole + nice_fraction`, so every machine
    runs at 100% except possibly one at 50%, 25%, 33.33% and so on.

    The requested amount is treated as a **ceiling**: producing a little less
    is fine if it buys tidy numbers, so the objective maximises output up to
    the target rather than pinning it there.
    """
    used = np.where(xs > EPS)[0]
    if used.size == 0 or used.size > EASY_NUMBERS_MAX_RECIPES:
        return xs                                  # too big to stay responsive

    hi_b = np.ceil(xs[used]) + 1
    fracs = [f for f, _ in NICE_FRACTIONS]
    costs = [c for _, c in NICE_FRACTIONS]
    U, K, nraw = used.size, len(fracs), m.n_raw
    # [b_j | z_jk | free_j | u_j | e]
    Z0, F0, U0 = U, U + U * K, U + U * K + U
    nb = U + U * K + U + U
    width = nb + nraw

    # Balance: sum_j net[:,j] * (b_j + sum_k q_k z_jk + free_j) + e
    A = np.zeros((len(m.items), width))
    for a, j in enumerate(used):
        A[:, a] = m.net[:, j]
        for k in range(K):
            A[:, Z0 + a * K + k] = m.net[:, j] * fracs[k]
        A[:, F0 + a] = m.net[:, j] * FREE_STEP
    for k2, it in enumerate(m.raw_items):
        A[m.item_ix[it], nb + k2] = 1.0

    # At most one fractional machine per recipe, and a free fraction counts as
    # that one. Insisting every stage sit on the lattice makes an exact balance
    # impossible, and the fix for an unbalanced stage is to tweak that stage -
    # not to overproduce everywhere else to make the numbers look tidy.
    pick = np.zeros((U, width))
    for a in range(U):
        pick[a, Z0 + a * K: Z0 + (a + 1) * K] = 1.0
        pick[a, U0 + a] = 1.0

    # free_j <= u_j  (a free fraction may only be used if its flag is set)
    freecap = np.zeros((U, width))
    for a in range(U):
        freecap[a, F0 + a] = FREE_STEP
        freecap[a, U0 + a] = -1.0

    targets = {it: v for it, v in demand.items() if v > EPS}
    lb = np.zeros(len(m.items))
    ub = np.full(len(m.items), np.inf)
    for it, v in targets.items():
        if it in m.item_ix:
            ub[m.item_ix[it]] = v                  # the amount is a ceiling

    # Stage 1 already worked out which leftover fluids are unavoidable. Snapping
    # to nice numbers must not introduce new ones, so cap each fluid at whatever
    # stage 1 settled on - otherwise rounding a refinery up quietly hands back
    # the Heavy Oil Residue this mode exists to avoid.
    dem_vec = _demand_vector(m, demand)
    _, e_base = _trim(m, dem_vec, xs)
    base_surplus = m.net @ xs
    for k2, it in enumerate(m.raw_items):
        base_surplus[m.item_ix[it]] += e_base[k2]
    for i in range(len(m.items)):
        if m.is_fluid[i]:
            allowed = max(demand.get(m.items[i], 0.0),
                          float(base_surplus[i]) + 1e-6)
            ub[i] = min(ub[i], allowed)

    # Two attempts. First demand an exact balance - nothing produced that is
    # not consumed - which is what kills the "constructor making 13 spare iron
    # plates" kind of leftover. Output is free to fall to whatever the nice
    # numbers can balance exactly, which is the point. Only if that is
    # genuinely impossible (a recipe whose co-product nobody needs, e.g.
    # Alumina Solution always yielding Silica) do we allow surplus through,
    # because that is just how the game works.
    # An item may only be left over if some recipe in the plan makes it
    # *alongside* something else (Alumina Solution always yielding Silica).
    # A leftover from a single-product recipe is just a machine running too
    # hard, which is the thing to stamp out.
    coproduct = set()
    for j in used:
        rc = m.recipe_ids[j]
        prods = m.data["recipes"][rc]["products"]
        if len(prods) > 1:
            for pr in prods:
                coproduct.add(pr["item"])
    strict_ub = ub.copy()
    for i in range(len(m.items)):
        if m.items[i] not in targets and m.items[i] not in coproduct:
            strict_ub[i] = STRICT_SURPLUS_TOL      # no unnecessary leftovers

    base_cons = [LinearConstraint(pick, lb=-np.inf, ub=1.0),
                 LinearConstraint(freecap, lb=-np.inf, ub=0.0)]
    attempts = [
        [LinearConstraint(A, lb=lb, ub=strict_ub)] + base_cons,
        [LinearConstraint(A, lb=lb, ub=ub)] + base_cons,
    ]

    # Maximise output up to the ceiling, then prefer tidy fractions, then fewer
    # machines. Scaled by the target so the shortfall term dominates.
    # Scale the output reward against the total cost this program could rack up.
    # A fixed pull is not enough: under a strict balance, building nothing is
    # free, so if the chain costs more than the reward the solver just makes
    # nothing - which is how "5 Computers" came back as a single idle machine.
    worst_cost = U * (EASIEST_MACHINE_WEIGHT * float(np.max(hi_b) + 1)
                      + FREE_FRACTION_COST + max(costs)) + 1.0
    pull = OUTPUT_PULL * worst_cost

    c = np.zeros(width)
    for it, v in targets.items():
        if it in m.item_ix:
            c -= A[m.item_ix[it]] / max(v, 1.0) * pull
    for a in range(U):
        c[a] += EASIEST_MACHINE_WEIGHT             # a whole machine still costs
        for k in range(K):
            c[Z0 + a * K + k] += costs[k]
        c[U0 + a] += FREE_FRACTION_COST            # an awkward clock, last resort
    c[nb:] += m.raw_machines * 0.01
    c[:nb] += _surplus_cost_cols(m, A[:, :nb])     # keep byproducts down

    lo = np.zeros(width)
    hi = np.concatenate([hi_b, np.ones(U * K),
                         np.full(U, 1.0 / FREE_STEP - 1), np.ones(U),
                         m.raw_limits])
    budget = 10 if depth == "absolute" else 5

    def score(vec_x):
        """Worst fraction of any target this plan actually delivers."""
        _, ve = _trim(m, dem_vec, vec_x)
        prod = m.net @ vec_x
        for k2, it in enumerate(m.raw_items):
            prod[m.item_ix[it]] += ve[k2]
        return min((min(v, float(prod[m.item_ix[it]])) / max(v, EPS)
                    for it, v in targets.items() if it in m.item_ix),
                   default=0.0)

    # g_j is integer: the free fraction is a whole number of 0.01% clock steps.
    integ = np.concatenate([np.ones(U), np.ones(U * K),
                            np.ones(U), np.ones(U), np.zeros(nraw)])
    # Stage 1's plan is the bar: tidy numbers are worthless if the factory
    # makes less (or nothing). A near-zero solution is always feasible under a
    # strict balance, so accepting the first non-empty answer was letting one
    # idle machine through as a valid plan for 5 Computers.
    results = []
    for cons in attempts:
        r = milp(c=c, constraints=cons, integrality=integ,
                 bounds=Bounds(lo, hi), options={"time_limit": budget})
        if not _usable(r):
            results.append(None)
            continue
        out = np.zeros(m.n_r)
        for a, j in enumerate(used):
            out[j] = (r.x[a]
                      + sum(fracs[k] * r.x[Z0 + a * K + k] for k in range(K))
                      + r.x[F0 + a] * FREE_STEP)
        results.append(out if out.sum() > EPS else None)
        # The strict plan makes nothing spare. Take it as soon as it delivers a
        # worthwhile amount, even a little under the target - trimming the rate
        # is the wanted answer, overproducing to hit the number exactly is not.
        if cons is attempts[0] and results[0] is not None                 and score(results[0]) >= STRICT_MIN_SCORE:
            return results[0]

    # Otherwise fall back to whichever plan actually makes the most. A lattice
    # plan is preferred even below stage 1's output: you can always scale a
    # working, dialable layout up later, whereas raw counts like 78.1481% are
    # unusable however much they produce.
    ok, tot = nice_fraction_stats(xs)
    xs_dialable = (tot == 0 or ok == tot)
    floor = (score(xs) - 1e-6) if xs_dialable else NICE_MIN_SCORE
    best, best_score = None, floor
    for out in results:
        if out is None:
            continue
        sc = score(out)
        if sc > best_score:
            best, best_score = out, sc
    return best if best is not None else xs


def _surplus_cost_cols(m: Model, cols):
    """Byproduct penalty projected onto an arbitrary column basis."""
    w = np.where(m.is_fluid, 0.02, 0.004)
    return w @ cols


def _optimise(m: Model, demand: dict, mode: str, depth: str, time_limit=None):
    """
    Returns (machine_counts, extraction, proven).

    `proven` is False when an integer solve was wanted but the solver ran out of
    time and fell back to a fractional answer, so the UI can say the figure is
    approximate. It is returned rather than stashed on the function object -
    that was process-global state, and two users solving at once would
    overwrite each other's flag.
    """
    flag = {"exact": True}
    block = _balance_rows(m)
    dem = _demand_vector(m, demand)
    lo, hi = _bounds(m)
    bnds = list(zip(lo, hi))

    # Extraction is free in every objective, so a bare LP happily mines every
    # resource up to its cap and dumps the surplus as phantom byproducts.
    # Solve in two stages: optimise the mode objective, then - holding that
    # objective at its optimum - mine as little as possible.
    extraction_cost = np.concatenate([np.zeros(m.n_r), m.raw_machines])

    def plain_lp(cost):
        res = linprog(cost, A_ub=-block, b_ub=-dem, bounds=bnds, method="highs")
        if not res.success:
            raise ValueError("Targets are not achievable within the resource "
                             "limits you set.")
        if not np.any(cost):                      # nothing to preserve
            return res.x[:m.n_r], res.x[m.n_r:]
        best = float(np.dot(cost, res.x))
        cap = np.vstack([-block, cost[None, :]])
        rhs = np.concatenate([-dem, [best + abs(best) * 1e-6 + 1e-6]])
        ref = linprog(extraction_cost, A_ub=cap, b_ub=rhs,
                      bounds=bnds, method="highs")
        x = ref.x if ref.success else res.x
        return x[:m.n_r], x[m.n_r:]

    if mode == "maximise":
        # Output is already pinned at its ceiling by the level solve, so this
        # mode is about *resource efficiency*: the same output for less input.
        # Raw draw is weighted by scarcity (share of that resource's whole-map
        # allowance) so it spends the plentiful stuff and protects the tight
        # stuff, rather than treating a unit of uranium like a unit of iron.
        scarcity = np.zeros(m.n_raw)
        for k in range(m.n_raw):
            cap = m.raw_limits[k]
            scarcity[k] = 0.0 if not np.isfinite(cap) else 1.0 / max(cap, 1.0)
        # 1.0 per unit drawn is the objective; scarcity and machines only ever
        # separate plans that draw the same total.
        cost = np.concatenate([np.full(m.n_r, MACHINE_TIEBREAK),
                               np.ones(m.n_raw) + scarcity * 1e-3])
        return (*plain_lp(cost), flag["exact"])

    if mode == "least_power":
        return (*plain_lp(np.concatenate([m.power + MACHINE_TIEBREAK, m.raw_power])), flag["exact"])

    if mode == "least_machines":
        # What a player actually places is ceil(x) buildings per recipe - the
        # last one underclocked. Optimise exactly that: `x` stays continuous
        # (so underclocking is allowed) and an integer building count `b >= x`
        # carries the objective.
        #
        # The two obvious shortcuts are both wrong, and between them they are
        # why Absolute could come out worse than Quick:
        #   - minimising fractional Sum(x) and then reporting Sum(ceil(x))
        #     reports a plan that does not balance once rounded up;
        #   - forcing x itself to be integer forbids underclocking altogether,
        #     which needs strictly more buildings.
        n = m.n_var
        nb, nr = m.n_r, m.n_raw
        width = n + nb + nr                       # [x, e, b_recipe, b_extractor]

        A = np.hstack([block, np.zeros((block.shape[0], nb + nr))])
        link = np.zeros((nb + nr, width))
        for j in range(nb):                       # x_j - b_j <= 0
            link[j, j] = 1.0
            link[j, n + j] = -1.0
        for k in range(nr):                       # e_k - rate_k * br_k <= 0
            rate = 1.0 / m.raw_machines[k] if m.raw_machines[k] > 0 else BIG_M
            link[nb + k, m.n_r + k] = 1.0
            link[nb + k, n + nb + k] = -rate
        cons = [LinearConstraint(A, lb=dem, ub=np.inf),
                LinearConstraint(link, lb=-np.inf, ub=0.0)]

        c = np.concatenate([np.zeros(n), np.ones(nb + nr)])
        lo2 = np.concatenate([lo, np.zeros(nb + nr)])
        hi2 = np.concatenate([hi, np.full(nb + nr, np.inf)])
        integrality = np.concatenate([np.zeros(n), np.ones(nb + nr)])

        abs_budget = time_limit or ABSOLUTE_TIME_LIMIT
        opts = ({"time_limit": abs_budget} if depth == "absolute"
                else {"time_limit": 4, "mip_rel_gap": 0.05})
        r = milp(c=c, constraints=cons, integrality=integrality,
                 bounds=Bounds(lo2, hi2), options=opts)
        if not _usable(r):
            flag["exact"] = False
            return plain_lp(np.concatenate([np.ones(m.n_r), m.raw_machines]))
        flag["exact"] = bool(r.success) and depth == "absolute"
        return (*_trim(m, dem, r.x[:m.n_r]), flag["exact"])

    if mode in ("least_steps", "easiest"):
        # Binary indicator per recipe: y_r = 1 if that recipe is used at all.
        n, nb = m.n_var, m.n_r
        raw_tie = m.raw_machines * 0.01           # discourage absurd extraction

        if mode == "least_steps":
            A = np.hstack([block, np.zeros((block.shape[0], nb))])
            link = np.zeros((nb, n + nb))         # x_r - M.y_r <= 0
            for j in range(nb):
                link[j, j] = 1.0
                link[j, n + j] = -BIG_M
            cons = [LinearConstraint(A, lb=dem, ub=np.inf),
                    LinearConstraint(link, lb=-np.inf, ub=0.0)]
            lo2 = np.concatenate([lo, np.zeros(nb)])
            hi2 = np.concatenate([hi, np.ones(nb)])
            c = np.concatenate([np.full(m.n_r, MACHINE_TIEBREAK), raw_tie,
                                np.ones(nb)])
            integrality = np.concatenate([np.zeros(n), np.ones(nb)])
        else:
            # "Easiest": fewest distinct steps, whole machines (a count like
            # 557.15 means hunting a 15% clock on the last one), and as few
            # leftovers as possible.
            #
            # A leftover fluid is a pain regardless of size - you cannot sink it,
            # so 5/min needs the same pipework as 500/min. Charging per unit
            # therefore reads the problem wrong and contorts the chain to shave
            # volume. Each fluid instead gets one binary and one flat cost worth
            # a few steps; solids keep a token per-unit nudge as a tie-break.
            fl = [i for i in range(len(m.items)) if m.is_fluid[i]]
            nf = len(fl)
            width = n + nb + nf
            A = np.hstack([block, np.zeros((block.shape[0], nb + nf))])
            link = np.zeros((nb + nf, width))
            for j in range(nb):                   # x_r - M.y_r <= 0
                link[j, j] = 1.0
                link[j, n + j] = -BIG_M
            fluid_rhs = np.zeros(nf)
            for q, i in enumerate(fl):            # surplus_i - M.f_i <= demand_i
                link[nb + q, :n] = block[i]
                link[nb + q, n + nb + q] = -BIG_M
                fluid_rhs[q] = demand.get(m.items[i], 0.0)
            cons = [LinearConstraint(A, lb=dem, ub=np.inf),
                    LinearConstraint(link, lb=-np.inf,
                                     ub=np.concatenate([np.zeros(nb), fluid_rhs]))]
            lo2 = np.concatenate([lo, np.zeros(nb + nf)])
            hi2 = np.concatenate([hi, np.ones(nb + nf)])
            c = np.concatenate([np.full(m.n_r, EASIEST_MACHINE_WEIGHT), raw_tie,
                                np.ones(nb), np.full(nf, FLUID_BYPRODUCT_COST)])
            c[:n] += _surplus_cost(m, solid=0.12, fluid=0.0)
            # Machine counts stay continuous *here*. Making ~290 of them integer
            # on top of the recipe and fluid binaries pushed the program past
            # what HiGHS could prove in 30s, and it returned a much worse
            # incumbent. Whole machines are restored in a second pass below,
            # over the handful of recipes this one actually selects.
            integrality = np.concatenate([np.zeros(n), np.ones(nb + nf)])
        abs_budget = time_limit or ABSOLUTE_TIME_LIMIT
        opts = ({"time_limit": abs_budget} if depth == "absolute"
                else {"time_limit": 6, "mip_rel_gap": 0.05})
        r = milp(c=c, constraints=cons, integrality=integrality,
                 bounds=Bounds(lo2, hi2), options=opts)
        if not _usable(r):
            # Nothing feasible at all - fall back so the user still gets a chain.
            flag["exact"] = False
            return plain_lp(np.concatenate([np.ones(m.n_r), np.zeros(m.n_raw)]))
        xs = r.x[:m.n_r]
        if not r.success:
            flag["exact"] = False         # ran out of time before proving it
        if mode == "easiest":
            xs = _easy_numbers(m, demand, xs, depth)
        return (*_trim(m, dem, xs), flag["exact"])

    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Emit a solver-shaped result (same keys the share solver returns)
# ---------------------------------------------------------------------------

def _to_solver_result(m: Model, x, e, demand: dict):
    out = {}
    for j, rc in enumerate(m.recipe_ids):
        if x[j] > EPS:
            out[f"{rc}@100#{m.machine_of[rc]}"] = float(x[j])
    for k, it in enumerate(m.raw_items):
        if e[k] > EPS:
            out[f"{it}#Mine"] = float(e[k])
    for it, amt in demand.items():
        if amt > EPS:
            out[f"{it}#Product"] = float(amt)

    produced = m.net @ x
    for i, it in enumerate(m.items):
        if it in demand:
            continue
        extra = produced[i] + (e[m.raw_ix[it]] if it in m.raw_ix else 0.0)
        if extra > 1e-4:
            out[f"{it}#Byproduct"] = float(extra)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def calculate(targets, mode="least_power", depth="quick", world=None,
              blocked_recipes=(), blocked_machines=(), name=None,
              resource_limits=None, least_resources="off", time_limit=None,
              allow_reduction=True):
    """
    Solve and return {meta, nodes, edges, layers, stats} ready for the UI.

    `least_resources` is "off" | "soft" | "hard" and narrows which raw resources
    may be used. It runs *after* the user's own resource choices and limits, and
    *before* the optimise-for mode: the chosen resources keep their full limits
    and the rest are treated as if unticked.
    """
    import ShareCodeResolver as SCR

    m = build_model(world, blocked_recipes, blocked_machines, resource_limits)

    # Block every machine and there is nothing left to solve with. Without this
    # the empty program reaches SciPy and comes back as "`c` must be a
    # one-dimensional array of finite numbers with at least one element", which
    # tells the user nothing about what they actually did.
    if m.n_r == 0:
        raise ValueError("No recipes are available - you have blocked every "
                         "machine or recipe that could make this.")

    # A negative order is not a smaller order, it is a request to consume the
    # item out of thin air, and the balance rows will happily oblige. Treat it
    # as nothing asked for.
    unknown = [t.get("item") for t in targets
               if t.get("item") and t["item"] not in m.item_ix
               and t["item"] not in m.data["items"]]
    if unknown:
        raise ValueError("Unknown item: " + ", ".join(sorted(set(unknown))))

    exact = {t["item"]: float(t.get("amount") or 0)
             for t in targets if t.get("type") == "exact"
             and float(t.get("amount") or 0) > 0}
    maximise = [t["item"] for t in targets if t.get("type") == "maximize"]
    linked = [t["item"] for t in targets if t.get("type") == "linked"]

    dropped = []
    if least_resources in ("soft", "hard"):
        # Choosing the resource set needs a demand to test against, but the
        # level of a MAX target is not known until we solve. Require a nominal
        # trickle of each so the set has to be able to make everything, then
        # maximise properly inside the set we picked.
        probe = dict(exact)
        for it in list(maximise) + list(linked):
            probe.setdefault(it, 1.0)
        if probe:
            keep = choose_resources(m, probe, mode, least_resources,
                                    maximise_items=list(maximise) + list(linked))
            dropped = [m.raw_items[k] for k in range(m.n_raw)
                       if not keep[k] and m.raw_limits[k] > 0]
            m = _restrict(m, keep)

    levels = _solve_levels(m, exact, maximise) if maximise else {}
    if linked:
        # Linked targets share out what is left after the exact ones (and any
        # ordinary MAX ones) have been settled.
        base = dict(exact)
        base.update({k: v for k, v in levels.items() if v > EPS})
        levels.update(_solve_levels_linked(m, base, linked))
    if mode == "easiest":
        levels = {k: _nice(v) for k, v in levels.items()}

    demand = dict(exact)
    # Never drop a maximised target for coming out small - asking to maximise an
    # item is still asking for it. Below the floor it is pinned to the floor so
    # it appears in the chain rather than silently disappearing.
    for k, v in levels.items():
        demand[k] = max(float(v), MIN_MAXIMISE_RATE)
    if not demand:
        raise ValueError("Add at least one target with an amount.")

    # A maximised target may give a little ground if that buys a better plan -
    # often it is the last couple of per cent that forces an awkward recipe in.
    x, e, integral, demand = _optimise_flexible(
        m, demand, set(levels) if allow_reduction else set(),
        mode, depth, time_limit)
    x = _prune_noise(m, x, e, demand)

    # Each resource needs its own extractors, so round up per resource. Taking
    # ceil() of the combined fraction under-reports whenever more than one
    # resource is tapped, and made "least machines" look like it had found a
    # smaller factory than it really had.
    extractors = int(sum(math.ceil(m.raw_machines[k] * e[k] - EPS)
                         for k in range(len(e)) if e[k] > EPS))

    # "easiest" is allowed to come in under the requested amount, so report what
    # the plan actually makes - not what was asked for. Reporting the request
    # was hiding plans that under-produced, or produced nothing at all.
    surplus = (m.net @ x)
    for k, it in enumerate(m.raw_items):
        surplus[m.item_ix[it]] += e[k]

    achieved = {}
    for it, want in demand.items():
        got = float(surplus[m.item_ix[it]]) if it in m.item_ix else 0.0
        achieved[it] = max(0.0, min(want, got))

    byproducts = {}
    for i, it in enumerate(m.items):
        extra = surplus[i] - achieved.get(it, 0.0)
        if extra > STRICT_SURPLUS_TOL and it not in demand:
            byproducts[it] = round(float(extra), 3)

    solver_result = _to_solver_result(m, x, e, achieved)
    nodes, edges = SCR._build_graph(solver_result, m.data)
    layers = SCR._topological_layers(nodes, edges)

    return {
        "meta": {
            "name": name or "Calculated production",
            "mode": mode,
            "depth": depth,
            "least_resources": least_resources,
            "calculated": True,
        },
        "nodes": nodes,
        "edges": edges,
        "layers": layers,
        "stats": {
            "power_mw":      round(float(np.dot(m.power, x)
                                          + np.dot(m.raw_power, e)), 2),
            "machines":      int(sum(math.ceil(v - EPS) for v in x if v > EPS)
                                 + extractors),
            "machines_frac": round(float(sum(x) + np.dot(m.raw_machines, e)), 3),
            "extractors":    extractors,
            "steps":         int(sum(1 for v in x if v > EPS)),
            "exact":         bool(integral),
            "resource_types": int(sum(1 for k, v in enumerate(e)
                                      if v > EPS and np.isfinite(m.raw_limits[k]))),
            "dropped_resources": dropped,
            "nice_machines":  nice_fraction_stats(x)[0],
            "machine_groups": nice_fraction_stats(x)[1],
            "byproducts":     len(byproducts),
            # Total leftover per minute. "Easiest" can leave a trickle (a few
            # hundredths) because machine counts are quantised, so report the
            # amount as well as the count - a silent 0.02/min looks like a
            # broken promise otherwise.
            "byproduct_total": round(float(sum(byproducts.values())), 3),
            "byproduct_fluid_total": round(float(sum(
                v for k, v in byproducts.items()
                if m.data["items"].get(k, {}).get("fluid"))), 3),
            "byproduct_fluids": int(sum(1 for it in byproducts
                                        if m.data["items"].get(it, {}).get("fluid"))),
            "byproduct_detail": byproducts,
            "outputs":       {k: round(v, 4) for k, v in achieved.items()},
            "requested":     {k: round(v, 4) for k, v in demand.items()},
            "raw":           {m.raw_items[k]: round(float(v), 4)
                              for k, v in enumerate(e) if v > EPS},
        },
    }


# ---------------------------------------------------------------------------
# ULTRA - exhaustive portfolio search
# ---------------------------------------------------------------------------

ULTRA_PER_CANDIDATE = 45          # seconds of solver time per combination tried

# Ceiling on solver threads across EVERY ultra job at once.
#
# Each job sizes its own pool, so three jobs used to mean three pools running
# flat out - 24 solves on a 24-core machine, which starves the web server and
# makes the site look broken while a search is on. The gate is process-wide, so
# extra jobs share these slots rather than adding to them, and a quarter of the
# machine is always left for serving pages.
ULTRA_MAX_CONCURRENT_SOLVES = max(2, ((os.cpu_count() or 4) * 3) // 4)
_ultra_gate = threading.BoundedSemaphore(ULTRA_MAX_CONCURRENT_SOLVES)

# Second pass: how many of the leading combinations get re-solved, and how much
# longer they are given. The sweep deliberately rushes every candidate, so a
# good one can place badly purely for want of time.
ULTRA_REFINE_COUNT = 24
ULTRA_REFINE_FACTOR = 12.0

# How deep the banned-resource sweep goes, as chosen in the UI. Each extra
# level adds C(12, depth) subsets per variety, so the cost climbs steeply and
# the returns do not - a plan that needs five specific resources gone is rare.
ULTRA_DEPTH_MIN = 3
ULTRA_DEPTH_MAX = 5
ULTRA_DEPTH_DEFAULT = 4


def _client_payload(res):
    """The shape the browser wants, with layers flattened to node ids."""
    return {
        "meta":   res["meta"],
        "nodes":  res["nodes"],
        "edges":  res["edges"],
        "layers": [[n["id"] for n in layer] for layer in res["layers"]],
        "stats":  res["stats"],
        "cached": False,
    }


def _ultra_worker(payload, queue, cancel_ev):
    """
    Child-process entry point. Streams the same three things the in-process
    version reported, as messages instead of callbacks.
    """
    try:
        def progress(checked, total, best_stats, stage=None):
            queue.put(("progress", checked, total, best_stats, stage))

        def on_best(result):
            queue.put(("best", _client_payload(result)))

        result = ultra_search(progress=progress, on_best=on_best,
                              cancelled=cancel_ev.is_set, **payload)
        queue.put(("done", _client_payload(result) if result else None))
    except Exception as exc:                        # noqa: BLE001
        queue.put(("error", f"{exc}"))


def start_ultra_process(payload):
    """Spawn a search. Returns (process, queue, cancel_event)."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")                   # Windows has no fork
    queue = ctx.Queue()
    cancel_ev = ctx.Event()
    proc = ctx.Process(target=_ultra_worker,
                       args=(payload, queue, cancel_ev), daemon=True)
    proc.start()
    return proc, queue, cancel_ev


def _prune_noise(m: Model, x, e, demand: dict):
    """
    Drop recipes running at a count too small to be real.

    Only where it is provably free: the plan still meets every demand and no
    item is left short. Anything actually load-bearing, however small, stays.
    """
    tiny = [i for i, v in enumerate(x) if EPS < v < NOISE_MACHINE_COUNT]
    if not tiny:
        return x

    def feasible(vec):
        surplus = m.net @ vec
        for k, it in enumerate(m.raw_items):
            surplus[m.item_ix[it]] += e[k]
        if (surplus < -STRICT_SURPLUS_TOL).any():
            return False
        # The slack has to scale with the demand. A flat 0.05/min tolerance is
        # larger than a maximised target sitting on its 0.01/min floor, so the
        # test read "surplus >= -0.04" and dropping the item's only producer
        # looked free - which zeroed the target outright.
        for it, want in demand.items():
            if it not in m.item_ix:
                continue
            slack = min(STRICT_SURPLUS_TOL, 0.05 * want)
            if surplus[m.item_ix[it]] < want - slack:
                return False
        return True

    out = np.array(x, dtype=float)
    for i in sorted(tiny, key=lambda k: x[k]):
        trial = out.copy()
        trial[i] = 0.0
        if feasible(trial):
            out = trial
    return out


def _plan_stats(m: Model, x, e, demand: dict, mode: str):
    """Just enough of the stats to judge a plan by its mode's own measure."""
    extractors = int(sum(math.ceil(m.raw_machines[k] * e[k] - EPS)
                         for k in range(len(e)) if e[k] > EPS))
    nice, groups = nice_fraction_stats(x)

    # Leftovers have to be counted here, not left at zero: "easiest" is judged
    # on them first, so a stub value made every plan look equally tidy and the
    # comparison silently fell through to the tie-breaks.
    surplus = (m.net @ x)
    for k, it in enumerate(m.raw_items):
        surplus[m.item_ix[it]] += e[k]
    byproducts = 0
    for i, it in enumerate(m.items):
        want = demand.get(it, 0.0)
        if it not in demand and surplus[i] > STRICT_SURPLUS_TOL:
            byproducts += 1
        elif it in demand and surplus[i] - want > STRICT_SURPLUS_TOL:
            byproducts += 1

    return {
        "machines": int(sum(math.ceil(v - EPS) for v in x if v > EPS) + extractors),
        "power_mw": float(np.dot(m.power, x) + np.dot(m.raw_power, e)),
        "steps":    int(sum(1 for v in x if v > EPS)),
        "nice_machines": nice,
        "machine_groups": groups,
        "byproducts": byproducts,
        "raw": {},
    }


def _worth_the_cut(base, cand, mode):
    """Is `cand` enough of an improvement to justify making less?"""
    if mode == "easiest":
        # Use the mode's own definition of better rather than a second opinion.
        # Ranking undialable clocks above leftovers here - while _metric ranks
        # them the other way - bought a plan with 15 leftovers to clear 12 bad
        # clocks, which is not the trade "easiest" is supposed to make.
        return _metric(cand, mode) < _metric(base, mode)
    b, c = _metric(base, mode), _metric(cand, mode)
    if not isinstance(b, (int, float)) or b <= 0:
        return False
    return (b - c) / b >= MAX_REDUCE_MIN_GAIN


def _optimise_flexible(m: Model, demand: dict, flexible: set, mode: str,
                       depth: str, time_limit):
    """
    Solve at the settled levels, then see whether easing the maximised targets
    buys a better plan. Returns (x, e, integral, demand_used).
    """
    tries = 1 + (len(MAX_REDUCE_STEPS) if flexible and mode != "maximise" else 0)

    # Share one budget across the attempts rather than spending a fresh one on
    # each. With no explicit limit the attempts each fell back to the full
    # absolute allowance, so a deep MAX chain went from 2s to 32s - four times
    # what "Absolute" is supposed to cost.
    budget = time_limit
    if budget is None and depth == "absolute" and tries > 1:
        budget = ABSOLUTE_TIME_LIMIT
    slot = (budget / tries) if budget else None

    x, e, integral = _optimise(m, demand, mode, depth, slot)
    if not flexible or mode == "maximise":
        return x, e, integral, demand

    base = _plan_stats(m, x, e, demand, mode)
    for cut in MAX_REDUCE_STEPS:
        eased = dict(demand)
        for it in flexible:
            if it in eased:
                eased[it] = max(MIN_MAXIMISE_RATE, eased[it] * (1.0 - cut))
        try:
            x2, e2, integral2 = _optimise(m, eased, mode, depth, slot)
        except Exception:                          # noqa: BLE001
            continue
        cand = _plan_stats(m, x2, e2, eased, mode)
        if _worth_the_cut(base, cand, mode):
            # Keep the demand at the eased level so the reported output is what
            # the plan actually makes, not what was originally settled.
            return x2, e2, integral2, eased
    return x, e, integral, demand


def _metric(stats, mode):
    """One mode's own figure of merit, lower being better."""
    if mode == "least_machines":
        return stats["machines"]
    if mode == "least_power":
        return stats["power_mw"]
    if mode == "least_steps":
        return stats["steps"]
    if mode == "easiest":
        groups = stats.get("machine_groups") or 0
        undialable = groups - (stats.get("nice_machines") or 0)
        return (stats["byproducts"], undialable, stats["steps"])
    return sum(stats["raw"].values())              # maximise: least input


def _ultra_score(stats, mode, priority=None):
    """
    Lower is better. Output comes first for every mode: a plan that makes more
    is never beaten by one that makes less, because the modes settle output
    before their own objective anyway.

    After that the chosen mode decides, and `priority` supplies the tie-breaks -
    so when two plans make the same per minute, the next preference in the list
    picks between them (e.g. same output and machines, take the one drawing
    less power).
    """
    out = sum(stats["outputs"].values())
    order = [mode] + [p for p in (priority or []) if p != mode]
    return (-round(out, 4), tuple(_metric(stats, p) for p in order))


# Resources whose recipes fan out into huge numbers of near-equivalent routes.
# The converters can turn almost any ore into almost any other, which multiplies
# the search space without usually improving the answer, and they only unlock
# right at the end of the game. Combinations that leave them out are tried first
# so a good plan surfaces early.
LATE_GAME_RESOURCES = ("Desc_SAM_C",)


def _ultra_candidates(resource_ids, base_lr, max_ban=None):
    """
    Combinations to try. Each is (variety, banned-resources).

    The user's own variety setting is honoured: picking "hard" means they want
    the fewest node types, so offering an "off" plan is not a better answer, it
    is a different question. Only "off" explores all three.

    Depth: subsets up to `max_ban` resources are enumerated exhaustively. The
    caller picks it (3-5); it applies to every variety in play, so the count is
    `depth-subsets x len(lrs)` and changing variety does not quietly change how
    deep the sweep goes.

    Ordering matters as much as coverage - the list is walked in order and the
    running best is shown live, so anything that bans a late-game resource is
    hoisted up front.
    """
    if base_lr == "hard":
        lrs = ("hard",)
    elif base_lr == "soft":
        lrs = ("soft", "hard")
    else:
        lrs = ("off", "soft", "hard")

    depth = ULTRA_DEPTH_DEFAULT if max_ban is None else int(max_ban)
    depth = max(ULTRA_DEPTH_MIN, min(ULTRA_DEPTH_MAX, depth))

    seen, out = set(), []

    def add(lr, banned):
        key = (lr, tuple(sorted(banned)))
        if key not in seen:
            seen.add(key)
            out.append(key)

    def subsets(items, k, prefix=(), start_at=0):
        if len(prefix) == k:
            yield prefix
            return
        for i in range(start_at, len(items)):
            yield from subsets(items, k, prefix + (items[i],), i + 1)

    for size in range(0, min(depth, len(resource_ids)) + 1):
        for lr in lrs:
            for combo in subsets(resource_ids, size):
                add(lr, combo)

    late = set(LATE_GAME_RESOURCES)
    # Stable sort: bans-a-late-game-resource first, then fewest bans.
    out.sort(key=lambda kv: (0 if (late & set(kv[1])) else 1, len(kv[1])))
    return out


def ultra_search(targets, mode="least_machines", world=None,
                 blocked_recipes=(), blocked_machines=(), resource_limits=None,
                 least_resources="off", name=None, priority=None,
                 progress=None, cancelled=None, workers=None, on_best=None,
                 ban_depth=None):
    """
    Try every sensible resource / variety combination and keep the best result.

    Candidates are independent, and HiGHS drops the GIL while it solves, so they
    run across a thread pool rather than one at a time. Single-threaded this was
    managing about seven combinations in five minutes on a late-game chain.

    `progress(checked, total, best_stats, stage)` is called as each candidate
    lands and `cancelled()` is polled throughout, so the caller can show how far
    along it is and stop at any point.

    `ban_depth` (3-5) sets how many resources may be banned at once. It is
    deliberately NOT called `depth` - that name already means quick/absolute in
    `calculate`, and the two are unrelated. It is the
    The cost is exponential and a deeper sweep is not guaranteed to find
    anything better, so it is the user's call.

    `on_best(result)` fires whenever a new leader appears. The caller keeps that
    snapshot so "Keep" can hand back the best-so-far instantly, rather than
    waiting for eight in-flight solves to drain first.
    """
    data = load_docs()
    base_limits = dict(resource_limits or {})

    usable = [cn for cn, info in data["resources"].items()
              if not info["unlimited"] and info["max"] > 0
              and base_limits.get(cn, info["max"]) > 0]
    usable.sort()

    combos = _ultra_candidates(usable, least_resources, ban_depth)
    total = len(combos)

    # Size the per-candidate budget to the problem. A wide late-game chain will
    # hit any limit on nearly every candidate, so a fixed 45s would mean the
    # search never finishes; a small chain solves in well under a second and
    # deserves the headroom to be proved optimal.
    probe = build_model(world, blocked_recipes, blocked_machines, base_limits)
    width = probe.n_r
    per_candidate = 30.0 if width < 120 else (15.0 if width < 220 else 8.0)

    if workers is None:
        workers = max(2, min(8, (os.cpu_count() or 4)))

    lock = threading.Lock()
    state = {"best": None, "key": None, "checked": 0, "ranked": [],
             "refined": 0, "best_seq": 0}

    # Callbacks are delivered through here, never straight from a worker.
    #
    # The workers pick up their snapshot under `lock` but call out afterwards,
    # so without this two threads can arrive in the wrong order and an older
    # leader overwrites a newer one. That is what made the reported best walk
    # backwards, and what let Keep hand back a plan that was not the best found.
    #
    # `emit_lock` is held across the callback on purpose: deciding "this one is
    # fresher" and delivering it have to be one step, or the same overtaking
    # happens a line later. It is a separate lock from `lock` so a slow callback
    # never blocks a solver, and no thread ever holds both in the other order.
    emit_lock = threading.Lock()
    emitted = {"seq": -1, "stats": None, "done": 0, "kept": -1}

    def emit(done, best_seq, best_stats, stage, leader=None, leader_seq=-1):
        with emit_lock:
            if best_seq > emitted["seq"]:
                emitted["seq"], emitted["stats"] = best_seq, best_stats
            if done > emitted["done"]:
                emitted["done"] = done
            if leader is not None and leader_seq > emitted["kept"]:
                emitted["kept"] = leader_seq
                if on_best:
                    on_best(leader)
            if progress:
                progress(emitted["done"], total, emitted["stats"], stage)

    def report(stage=None):
        with lock:
            done = state["checked"]
            seq = state["best_seq"]
            st = state["best"]["stats"] if state["best"] else None
        emit(done, seq, st, stage)

    def run_one(combo):
        lr, banned = combo
        if cancelled is not None and cancelled():
            return
        limits = dict(base_limits)
        for cn in banned:
            limits[cn] = 0
        result = None
        new_leader = None
        leader_seq = -1
        try:
            with _ultra_gate:
                # Checked again here: a cancel while queued on the gate should
                # not then start a fresh multi-second solve.
                if cancelled is not None and cancelled():
                    return
                # The sweep is comparing resource sets, so it skips the
                # give-a-little pass - four solves per candidate would make it
                # four times longer for a judgement the refinement redoes
                # properly anyway.
                result = calculate(targets, mode=mode, depth="absolute",
                                   world=world,
                                   blocked_recipes=blocked_recipes,
                                   blocked_machines=blocked_machines,
                                   resource_limits=limits, least_resources=lr,
                                   name=name, time_limit=per_candidate,
                                   allow_reduction=False)
        except ValueError:
            pass                                   # infeasible combination
        except Exception:                          # noqa: BLE001
            pass

        with lock:
            state["checked"] += 1
            if result is not None:
                key = _ultra_score(result["stats"], mode, priority)
                if state["best"] is None or key < state["key"]:
                    result["meta"]["ultra_from"] = {"least_resources": lr,
                                                    "banned": list(banned)}
                    state["best"], state["key"] = result, key
                    state["best_seq"] += 1
                    new_leader, leader_seq = result, state["best_seq"]
                state["ranked"].append((key, combo))
            done = state["checked"]
            best_seq = state["best_seq"]
            best_stats = state["best"]["stats"] if state["best"] else None

        emit(done, best_seq, best_stats, f"{lr}, {len(banned)} off",
             new_leader, leader_seq)

    report("starting")
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(run_one, c) for c in combos]
        try:
            for _ in futures.as_completed(pending):
                if cancelled is not None and cancelled():
                    for f in pending:
                        f.cancel()                 # queued ones never start
                    break
        except Exception:                          # noqa: BLE001
            pass

    # Second pass: the sweep gives every candidate the same modest budget, so a
    # promising one can look mediocre purely because it ran out of time. Re-solve
    # the leaders with a far longer budget to settle the order properly.
    leaders = sorted((r for r in state["ranked"] if r[1] is not None),
                     key=lambda kv: kv[0])[:ULTRA_REFINE_COUNT]
    if leaders and not (cancelled is not None and cancelled()):
        if progress:
            progress(state["checked"], total,
                     state["best"]["stats"] if state["best"] else None,
                     f"refining top {len(leaders)}")

        def refine(entry):
            _key, combo = entry
            if cancelled is not None and cancelled():
                return

            def done_one():
                # The sweep is over, so `checked` has stopped moving. Report the
                # refinement as its own count - at 24 candidates x 12x budget
                # this pass is long, and a stage string frozen on "refining top
                # 24" for that whole time looks like a hang.
                with lock:
                    state["refined"] += 1
                    n = state["refined"]
                    done = state["checked"]
                    seq = state["best_seq"]
                    snap = state["best"]["stats"] if state["best"] else None
                emit(done, seq, snap, f"refining {n}/{len(leaders)}")

            lr, banned = combo
            limits = dict(base_limits)
            for cn in banned:
                limits[cn] = 0
            try:
                with _ultra_gate:
                    if cancelled is not None and cancelled():
                        return
                    res = calculate(targets, mode=mode, depth="absolute",
                                    world=world,
                                    blocked_recipes=blocked_recipes,
                                    blocked_machines=blocked_machines,
                                    resource_limits=limits, least_resources=lr,
                                    name=name,
                                    time_limit=per_candidate * ULTRA_REFINE_FACTOR)
            except Exception:                      # noqa: BLE001
                done_one()
                return
            key = _ultra_score(res["stats"], mode, priority)
            with lock:
                if state["best"] is None or key < state["key"]:
                    res["meta"]["ultra_from"] = {"least_resources": lr,
                                                 "banned": list(banned),
                                                 "refined": True}
                    state["best"], state["key"] = res, key
                    state["best_seq"] += 1
                    leader, leader_seq = res, state["best_seq"]
                else:
                    leader, leader_seq = None, -1
                done = state["checked"]
                best_seq = state["best_seq"]
                best_stats = state["best"]["stats"] if state["best"] else None
            if leader is not None:
                emit(done, best_seq, best_stats, None, leader, leader_seq)
            done_one()

        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(refine, leaders))

    best = state["best"]
    if best is not None:
        best["meta"]["ultra"] = True
        best["stats"]["ultra_checked"] = state["checked"]
        best["stats"]["ultra_total"] = total
        best["stats"]["ultra_workers"] = workers
        best["stats"]["ultra_refined"] = len(leaders)
    return best
