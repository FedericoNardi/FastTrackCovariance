from pathlib import Path
import argparse
import json
import os
import subprocess
import numpy as np
import uproot
import matplotlib.pyplot as plt
from skopt.space import Real
from skopt import gp_minimize
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace
from threadpoolctl import threadpool_limits
import time
import threading

ROOT_DIR = Path(__file__).resolve().parent          # repo root: ROOT macros are loaded from here
OPT_DIR  = ROOT_DIR / "opt"

GEO_BASE         = ROOT_DIR / "GeoCLD.txt"
GEO_OPT          = OPT_DIR / "GeoOPT.txt"
METRICS_FILE     = OPT_DIR / "metrics.root"
RUNMETRICS_C     = "RunMetrics.cc"
LOADALL_C        = "LoadAll.c"

_cache = {}
_baseline = None   # baseline resolutions, filled by baseline_metrics()
_baseline_lock = threading.Lock()   # parallel evaluations must not compute it twice

# ============================================================================
# COLLIDER MODE (set by initialize_optimization_config)
#   MC: muon collider. The tungsten nozzles are present: barrels, disks and
#       tracks must stay outside them, and they replace the beam pipe as the
#       innermost passive element (the PIPE layer is dropped).
#   HC: hadron collider. No nozzle; the beam pipe is the innermost passive layer.
# ============================================================================
COLLIDERS = ("MC", "HC")
COLLIDER = "MC"

def has_nozzle():
    return COLLIDER == "MC"

def geo_baseline_file():
    """Baseline geometry as used in the current collider mode."""
    return OPT_DIR / f"GeoBASE_{COLLIDER}.txt"

def metrics_baseline_file():
    return OPT_DIR / f"metrics_baseline_{COLLIDER}.root"

# ============================================================================
# TRACK GRID (passed to RunMetrics.cc)
# ============================================================================
N_THETA, THETA_MIN_DEG = 40, 10.0          # uniform in cos(theta) from THETA_MIN_DEG to 90 deg
N_PT, PT_MIN, PT_MAX   = 20, 0.5, 100.0    # log-spaced, GeV

# ============================================================================
# GEOMETRY CONSTANTS
# ============================================================================
DOUBLET_MAX_GAP = 0.005   # sensors of the same label/length closer than this move as one station (m)
MIN_GAP         = 0.002   # minimum radial clearance between a station and anything else (m)
# Interface between inner and outer tracker: the fixed passive layers lying
# between the outermost INTERFACE_LABELS[0] station and the innermost
# INTERFACE_LABELS[1] station (the ITK support tube) are surrounded by an empty
# band: no station may come closer than INTERFACE_GAP to them.
INTERFACE_LABELS = ("ITK", "OTK")
INTERFACE_GAP   = 0.020   # (m)
R_FRAC_LO, R_FRAC_HI = 0.70, 1.30   # station radius range relative to baseline
# Outer envelope (the calorimeter sits outside it). Set by
# initialize_optimization_config to the bounding box of the base geometry
# (beam pipe excluded): no barrel radius, barrel half-length or disk |z| may exceed it.
R_ABS_MAX       = 1.600   # max barrel radius (m)
CLEARANCE       = 0.001   # clearance to the nozzle and to disks (m)
L_MIN, L_ABS_MAX = 0.050, 2.000     # barrel half-length range (m); max set from the envelope
Z_FRAC_LO, Z_FRAC_HI = 0.70, 1.30   # disk |z| range relative to baseline
Z_ABS_MAX       = 2.500   # max disk |z| (m); set from the envelope

# ============================================================================
# LOSS CONSTANTS
# ============================================================================
FAIL_LOSS = 3.0           # returned for failed evaluations; the baseline scores 1.0

# ============================================================================
# TRACK-FINDING MODEL (placeholders, to be tuned to the real reconstruction)
# A track is found if it has at least SEED_HITS_MIN hits in the seed
# subsystems AND at least N_HITS_MIN measured hits in total. Every crossed
# measurement layer gives a hit with probability HIT_EFF[label], so the
# finding efficiency of each track is a smooth function of the geometry.
# Loss = efficiency-weighted resolution ratio + EFF_WEIGHT * (1 - eff/eff_baseline)
# ============================================================================
N_HITS_MIN    = 6
SEED_HITS_MIN = 3
SEED_LABELS   = ("VTX", "VTXDSK", "ITK", "ITKDSK")
HIT_EFF       = {"VTX": 0.98, "VTXDSK": 0.98, "ITK": 0.98, "ITKDSK": 0.98, "OTK": 0.98, "OTKDSK": 0.98}
HIT_EFF_DEFAULT = 0.98
EFF_WEIGHT    = 1.0       # 1% relative efficiency loss costs as much as 1% worse resolution

# ============================================================================
# RESOURCES: use at most CPU_FRACTION of the node, both for the parallel ROOT
# evaluations and for the threads of the GP fits (torch / BLAS)
# ============================================================================
CPU_FRACTION = 0.25

def cpu_cap():
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return max(1, int(n * CPU_FRACTION))
GP_NOISE  = 1e-6          # the objective is deterministic: fix a tiny noise variance (skopt)
GP_NOISE_REL = 1e-4       # same for BoTorch, as a fraction of the variance of the fitted outcome

# Thompson sampling in a trust region (TuRBO, Eriksson et al. 2019; BoTorch backend).
# Candidates are drawn in a box around the best point, scaled per parameter by
# the GP length scales. The box (edge TR_LENGTH_INIT in the unit cube) doubles
# after TR_SUCCESS_ROUNDS improving rounds, halves after enough failed rounds
# and is reset to TR_LENGTH_INIT once smaller than TR_LENGTH_MIN.
TS_CANDIDATES      = 5000
TR_LENGTH_INIT     = 0.8
TR_LENGTH_MIN      = 0.5 ** 7
TR_LENGTH_MAX      = 1.6
TR_SUCCESS_ROUNDS  = 3
TR_IMPROVEMENT     = 1e-3   # relative improvement of the best loss that counts as success

# ============================================================================
# GLOBAL CONFIGURATION - Initialized by initialize_optimization_config()
# ============================================================================
BASE_LAYERS = []
STATIONS = []          # all barrel measurement stations, sorted by radius
FIXED_RADII = []       # barrel layers that never move (beam pipe, support shells)
FREE_STATIONS = []     # the subset being optimized
N_FREE_BARREL = 0      # number of free barrel stations
FREE_BARREL_IDX = []   # layer indices moved by the free stations (sensors + attached supports)
DISK_STATIONS = []     # all +z disk stations (each mirrored to -z), sorted by z
FREE_DISK_STATIONS = []
N_FREE_DISK = 0       # number of free disk stations
THETA_DIM = 0

def read_metrics(metrics_path):
    """Read the metrics tree and check it is complete and finite."""
    with uproot.open(metrics_path) as f:
        arr = f["metrics"].arrays(library="np")
    n_expected = N_THETA * N_PT
    if len(arr["spt_rel"]) != n_expected:
        raise RuntimeError(f"{metrics_path} has {len(arr['spt_rel'])} entries, expected {n_expected}")
    for k in ("spt_rel", "sd0_um"):
        if not np.all(np.isfinite(arr[k])) or np.any(arr[k] <= 0):
            raise RuntimeError(f"{metrics_path} contains non-finite or non-positive {k}")
    return arr

def baseline_metrics():
    """Resolutions of the base geometry on the same track grid (computed once)."""
    global _baseline
    with _baseline_lock:
        if _baseline is None:
            write_layers(BASE_LAYERS, geo_baseline_file())
            run_root_metrics(geo_baseline_file(), metrics_baseline_file())
            base = read_metrics(metrics_baseline_file())
            base["eff_mean"] = float(np.mean(track_efficiencies(base, BASE_LAYERS)))
            _baseline = base
    return _baseline

def _pmf_at_least(probs, k):
    """P(at least k successes) for independent trials with success probabilities `probs`."""
    pmf = np.zeros(len(probs) + 1); pmf[0] = 1.0
    for p in probs:
        pmf[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
        pmf[0] *= (1 - p)
    return pmf

def track_efficiencies(arr, layers):
    """
    Finding efficiency of every track in a metrics tree (see TRACK-FINDING MODEL).
    `layers` is the geometry the metrics were computed with: the stored layer
    indices refer to it.
    """
    labels = [L["label"] for L in layers]
    eff = np.empty(len(arr["nmeas"]))
    for t, idx in enumerate(arr["mlay"]):
        seed = [HIT_EFF.get(labels[i], HIT_EFF_DEFAULT) for i in idx if labels[i] in SEED_LABELS]
        other = [HIT_EFF.get(labels[i], HIT_EFF_DEFAULT) for i in idx if labels[i] not in SEED_LABELS]
        ps, po = _pmf_at_least(seed, 0), _pmf_at_least(other, 0)
        tail_o = np.cumsum(po[::-1])[::-1]              # P(other >= m)
        e = 0.0
        for n_seed in range(SEED_HITS_MIN, len(seed) + 1):
            need = max(0, N_HITS_MIN - n_seed)
            e += ps[n_seed] * (tail_o[need] if need < len(tail_o) else 0.0)
        eff[t] = e
    return eff

def loss_components(metrics_path=METRICS_FILE, layers=None):
    """
    Loss and its parts for one metrics file, relative to the baseline geometry
    evaluated on the same tracks:

      res  = sum_t eff_t * r_t / sum_t eff_t,  r_t = (σpT/pT ratio + σd0 ratio) / 2
             (resolution ratios to the baseline, weighted by finding efficiency,
              so unfindable tracks do not dominate with huge resolutions)
      eff  = mean finding efficiency over the track grid
      loss = res + EFF_WEIGHT * (1 - eff / eff_baseline)

    The baseline scores exactly 1.0. Lower is better.
    `layers`: geometry the metrics belong to (default: read GEO_OPT).
    """
    if layers is None:
        layers = load_layers(GEO_OPT)
    arr = read_metrics(metrics_path)
    base = baseline_metrics()
    if not (np.allclose(arr["pt"], base["pt"]) and np.allclose(arr["theta_deg"], base["theta_deg"])):
        raise RuntimeError("track grid differs from the baseline metrics")

    r = 0.5 * (arr["spt_rel"] / base["spt_rel"]) + 0.5 * (arr["sd0_um"] / base["sd0_um"])
    eff = track_efficiencies(arr, layers)
    res = float(np.sum(eff * r) / np.sum(eff)) if np.sum(eff) > 0 else FAIL_LOSS
    e_mean = float(np.mean(eff))
    loss = res + EFF_WEIGHT * (1.0 - e_mean / base["eff_mean"])
    return {"loss": loss, "res": res, "eff": e_mean, "eff_base": base["eff_mean"]}

def compute_loss_from_metrics(metrics_path=METRICS_FILE, layers=None):
    """Scalar loss of a metrics file (see loss_components)."""
    return loss_components(metrics_path, layers)["loss"]

def nozzle_profile(x):
    m1 = 0.1763
    m2 = 0.08474
    q2 = 9.156*0.01 # meters
    return np.where(x<1., m1*x, m2*x+q2)

def nozzle_inverse(r):
    """Largest |z| at which the nozzle radius is still below r (inverse of nozzle_profile).
    Unbounded when there is no nozzle (HC)."""
    if not has_nozzle():
        return np.inf
    m1 = 0.1763
    m2 = 0.08474
    q2 = 9.156*0.01 # meters
    if r <= 0.0:
        return 0.0
    return r / m1 if r < m1 else (r - q2) / m2

def barrel_nozzle_penalty(theta, clearance=CLEARANCE, scale=0.01):
    """
    Compute penalty for barrel layers that violate nozzle clearance.

    Only checks the barrel layers moved by the optimizer (sensors and their
    attached supports), not fixed layers like the beam pipe which have
    different constraints. Zero by construction for lengths built by
    build_layers_from_theta; kept as a safety check.
    """
    if not has_nozzle():
        return 0.0
    layers = build_layers_from_theta(theta)
    penalty = 0.0

    for idx in FREE_BARREL_IDX:
        L = layers[idx]
        R_b   = L["rPos"]
        z_end = max(abs(L["xMin"]), abs(L["xMax"]))  # half-length
        r_noz = float(nozzle_profile(z_end)) + clearance
        v = r_noz - R_b

        if v > 1e-9:
            penalty += (v / scale) ** 2

    return penalty

def barrel_disk_overlap_penalty(theta, clearance=CLEARANCE, scale=0.01):
    """
    Compute penalty for moved barrel layers that intersect any disk layer.

    A barrel (radius R_b, half-length L_b) and a disk (plane z_d, radii
    [r_min, r_max]) overlap when |z_d| < L_b and r_min < R_b < r_max, each
    widened by the clearance. The violation is the smallest displacement that
    resolves it: shortening the barrel or moving it radially off the disk.
    Fixed disks are included, since free barrels can grow into them, and so
    are fixed barrel layers (support shells, not the beam pipe), since free
    disks can move into them.
    Zero by construction for lengths built by build_layers_from_theta; kept
    as a safety check.
    """
    layers = build_layers_from_theta(theta)
    penalty = 0.0

    disks = [L for L in layers if L["tyLay"] == 2]
    fixed = {i for i, L in enumerate(layers)
             if L["tyLay"] == 1 and L["label"] != "PIPE"
             and i not in {idx for s in STATIONS for idx, _ in s["members"]}}

    for idx in list(FREE_BARREL_IDX) + sorted(fixed):
        b = layers[idx]
        R_b = b["rPos"]
        L_b = max(abs(b["xMin"]), abs(b["xMax"]))

        for d in disks:
            axial  = (L_b + clearance) - abs(d["rPos"])
            radial = min(R_b - (d["xMin"] - clearance), (d["xMax"] + clearance) - R_b)
            if axial > 1e-9 and radial > 0.0:
                v = min(axial, radial)
                penalty += (v / scale) ** 2

    return penalty

def station_length_max(station, R, layers, clearance=CLEARANCE):
    """
    Longest half-length a station at reference radius R can have without
    entering the nozzle or crossing any disk in `layers`.

    The nozzle radius grows with |z|, so the innermost member is the binding
    one. A disk limits the length if any member's radius falls inside its
    radial extent (widened by the clearance), matching the overlap penalty.
    """
    radii = [R + dr for _, dr in station["members"]]
    L_max = min(L_ABS_MAX, nozzle_inverse(min(radii) - clearance))
    for d in layers:
        if d["tyLay"] != 2:
            continue
        if any(d["xMin"] - clearance < r < d["xMax"] + clearance for r in radii):
            L_max = min(L_max, abs(d["rPos"]) - clearance)
    return L_max

def length_from_fraction(f, L_max):
    """Map f in [0, 1] onto [L_MIN, L_max]."""
    return L_MIN + f * max(L_max - L_MIN, 0.0)

def score(theta, workdir=None):
    """
    Objective function for Bayesian optimization.
    workdir: directory for this evaluation's geometry, metrics and ROOT log
             (needed when several evaluations run in parallel); default opt/.
    theta: [R_1, ..., R_N, f_1, ..., f_N, Z_1, ..., Z_M]
           R is the reference radius of each free station, f its half-length
           as a fraction of the longest allowed one (see station_length_max).
    Returns: scalar loss value (lower is better, baseline = 1.0)
    """
    theta = np.asarray(theta, dtype=float)
    expected_len = 2 * N_FREE_BARREL + N_FREE_DISK
    if theta.shape[0] != expected_len:
        raise ValueError(f"theta length {theta.shape[0]} != {expected_len}")

    # Use higher precision for cache to avoid false misses
    key = tuple(theta.round(12))
    if key in _cache:
        print(f"[CACHE HIT] Returning cached loss: {_cache[key]:.4g}")
        return _cache[key]

    Nb = N_FREE_BARREL
    Nd = N_FREE_DISK
    R_b = theta[0:Nb]
    f_b = theta[Nb:2*Nb]
    Z_d = theta[2*Nb:2*Nb + Nd] if Nd > 0 else np.array([])

    # --- basic bounds (should be enforced by optimizer, but double-check) ---
    # Absolute physical limits
    R_abs_min, R_abs_max = 0.020, 2.000  # absolute barrel radius limits
    Z_abs_min, Z_abs_max = 0.01, 2.5     # disk z-range

    if (np.any(R_b < R_abs_min) or np.any(R_b > R_abs_max) or
        np.any(f_b < 0.0) or np.any(f_b > 1.0)):
        print(f"[OUT OF BOUNDS] R={R_b}, f={f_b}")
        _cache[key] = FAIL_LOSS
        return FAIL_LOSS

    if Nd > 0 and (np.any(Z_d < Z_abs_min) or np.any(Z_d > Z_abs_max)):
        print(f"[OUT OF BOUNDS] Z={Z_d}")
        _cache[key] = FAIL_LOSS
        return FAIL_LOSS

    # --- base tracking loss ---
    if workdir is None:
        geo_file, metrics_file, log_file = GEO_OPT, METRICS_FILE, None
    else:
        workdir = Path(workdir)
        geo_file, metrics_file, log_file = workdir / "GeoOPT.txt", workdir / "metrics.root", workdir / "root.log"
    try:
        layers = build_layers_from_theta(theta)
        write_layers(layers, geo_file)
        run_root_metrics(geo_file, metrics_file, log_file=log_file)
        parts = loss_components(metrics_file, layers)
        base_loss = parts["loss"]
    except Exception as e:
        print(f"[ERROR] Failed to compute metrics: {e}")
        _cache[key] = FAIL_LOSS
        return FAIL_LOSS

    # --- nozzle area forbidden (zero by construction) ---
    nozzle_penalty = barrel_nozzle_penalty(theta)

    # --- disk-barrel overlap forbidden (zero by construction) ---
    overlap_penalty = barrel_disk_overlap_penalty(theta)

    loss = base_loss + nozzle_penalty + overlap_penalty

    layers = build_layers_from_theta(theta)
    L_b = [layers[s["sensors"][0]]["xMax"] for s in FREE_STATIONS]
    print(
        f"R_b={[f'{r:.4f}' for r in R_b]}, L_b={[f'{l:.4f}' for l in L_b]} -> "
        f"res={parts['res']:.4f}, eff={parts['eff']:.4f} (base {parts['eff_base']:.4f}), "
        f"base={base_loss:.4f}, nozzle={nozzle_penalty:.4g}, "
        f"overlap={overlap_penalty:.4g}, total={loss:.4f}"
    )

    _cache[key] = loss
    return loss

def run_root_metrics(geo_file=GEO_OPT, metrics_file=METRICS_FILE, log_file=None):
    """
    Run RunMetrics.cc on geo_file. The output file is removed first so a failed
    run can never leave the previous geometry's metrics behind to be read.
    log_file: if given, ROOT's output goes there instead of the terminal.
    """
    metrics_file = Path(metrics_file)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.unlink(missing_ok=True)

    cmd = [
        "root",
        "-b",
        "-q",
        f'{LOADALL_C}("CLD")',         # compile library
        f'{RUNMETRICS_C}+("{geo_file}", "{metrics_file}", '
        f'{N_THETA}, {THETA_MIN_DEG}, {N_PT}, {PT_MIN}, {PT_MAX})',
    ]
    if log_file is None:
        print("Running ROOT:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=ROOT_DIR)
    else:
        with open(log_file, "w") as log:
            subprocess.run(cmd, check=True, cwd=ROOT_DIR, stdout=log, stderr=subprocess.STDOUT)

    if not metrics_file.exists():
        raise RuntimeError(f"ROOT exited cleanly but did not write {metrics_file}")

def draw_from_file(geom_file=GEO_OPT, out_file=None, title=None):
    """
    Draw the (z, r) layout of a geometry file.

    Active (measurement) layers are solid and coloured: barrels blue, disks
    red. Passive layers (supports, beam pipe) are grey and dashed; their line
    width grows with their material budget. The nozzle region is shaded
    (MC only).
    """
    layers = load_layers(geom_file)
    out_file = Path(out_file) if out_file else OPT_DIR / "geometry.png"
    z_lim, r_lim = 2.6, 1.7

    fig, (ax, axin) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [3, 1]})

    def draw(ax, z_max):
        # Nozzle (forbidden region, MC only)
        if has_nozzle():
            z = np.linspace(0.0, z_max, 200)
            for sign in (-1, 1):
                ax.fill_between(sign * z, 0.0, nozzle_profile(z), color="0.85", lw=0)
        for L in layers:
            active = L["flLay"] == 1
            x0 = L["thLay"] / L["rlLay"] if L["rlLay"] > 0 else 0.0   # fraction of X0
            if active:
                color, ls, lw = ("tab:blue" if L["tyLay"] == 1 else "tab:red"), "-", 1.8
            else:
                color, ls, lw = "0.35", "--", 0.8 + 150.0 * x0   # ~1.6 for 0.5% X0
            if L["tyLay"] == 1:   # barrel: z from xMin to xMax at r = rPos
                zmin, zmax = max(L["xMin"], -z_max), min(L["xMax"], z_max)
                ax.plot([zmin, zmax], [L["rPos"], L["rPos"]], color=color, ls=ls, lw=lw)
            else:                 # disk: r from xMin to xMax at z = rPos
                ax.plot([L["rPos"], L["rPos"]], [L["xMin"], L["xMax"]], color=color, ls=ls, lw=lw)

    draw(ax, z_lim)

    # Inner/outer tracker interface keep-out band
    iface = sorted(interface_radii(STATIONS, FIXED_RADII)) if STATIONS else []
    if iface:
        ax.fill_between([-Z_ABS_MAX, Z_ABS_MAX], iface[0] - INTERFACE_GAP, iface[-1] + INTERFACE_GAP,
                        color="tab:green", alpha=0.12, lw=0)

    # Tracker envelope (hard limit)
    ax.add_patch(plt.Rectangle((-Z_ABS_MAX, 0.0), 2 * Z_ABS_MAX, R_ABS_MAX, fill=False,
                               ec="tab:green", ls=":", lw=1.2))

    # Zoom on the vertex region (side panel)
    zi, ri = 0.35, 0.18
    draw(axin, zi)
    axin.set_xlim(-zi, zi)
    axin.set_ylim(0.0, ri)
    axin.set_xlabel("z [m]")
    axin.set_title("vertex region")
    ax.add_patch(plt.Rectangle((-zi, 0.0), 2 * zi, ri, fill=False, ec="0.4", lw=0.8))

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color="tab:blue", lw=1.8, label="active barrel"),
        Line2D([], [], color="tab:red", lw=1.8, label="active disk"),
        Line2D([], [], color="0.35", ls="--", lw=1.2, label="passive (width ~ X/X0)"),
        Line2D([], [], color="tab:green", ls=":", lw=1.2, label="envelope"),
        plt.Rectangle((0, 0), 1, 1, color="tab:green", alpha=0.12, label="interface keep-out"),
    ]
    if has_nozzle():
        handles.append(plt.Rectangle((0, 0), 1, 1, color="0.85", label="nozzle"))
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    ax.set_xlim(-z_lim, z_lim)
    ax.set_ylim(0.0, r_lim)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("r [m]")
    ax.set_title(f"{title or Path(geom_file).name}  [{COLLIDER}]")
    fig.tight_layout()
    fig.savefig(out_file, dpi=120)
    plt.close(fig)

def load_layers(geo_path=GEO_BASE):
    """Read GeoCLD.txt into a list of layer dicts."""
    layers = []
    with open(geo_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # tyLay LyLabl xMin xMax rPos thLay rlLay nmLay stU stL sgU sgL flLay
            layer = {
                "tyLay":  int(parts[0]),
                "label":  parts[1],
                "xMin":   float(parts[2]),
                "xMax":   float(parts[3]),
                "rPos":   float(parts[4]),
                "thLay":  float(parts[5]),
                "rlLay":  float(parts[6]),
                "nmLay":  int(parts[7]),
                "stLayU": float(parts[8]),
                "stLayL": float(parts[9]),
                "sgLayU": float(parts[10]),
                "sgLayL": float(parts[11]),
                "flLay":  int(parts[12]),
            }
            layers.append(layer)
    return layers

def build_stations(layers):
    """
    Group barrel layers into rigid stations.

    A station is one barrel measurement layer, or a doublet of measurement
    layers with the same label and length closer than DOUBLET_MAX_GAP. Each
    passive barrel layer (flLay=0, not the beam pipe) is attached to the
    nearest station with the same label and the same half-length, i.e. the
    support it belongs to. Passive layers with no such station (support shells,
    tubes) stay fixed.

    Returns (stations, fixed_radii): stations sorted by radius, each with
    'R0' (reference radius = innermost sensor), 'L0', 'sensors' and
    'members' as (layer index, radial offset from R0); fixed_radii are radii of
    the barrel layers that never move (beam pipe and unattached passives).
    """
    def half_length(L):
        return 0.5 * (L["xMax"] - L["xMin"])

    sensors = sorted(
        (i for i, L in enumerate(layers) if L["tyLay"] == 1 and L["flLay"] == 1),
        key=lambda i: layers[i]["rPos"],
    )

    stations = []
    for i in sensors:
        L = layers[i]
        if stations:
            prev = layers[stations[-1]["sensors"][-1]]
            if (L["label"] == prev["label"]
                    and np.isclose(half_length(L), half_length(prev))
                    and L["rPos"] - prev["rPos"] <= DOUBLET_MAX_GAP):
                stations[-1]["sensors"].append(i)
                continue
        stations.append({"label": L["label"], "R0": L["rPos"], "L0": half_length(L), "sensors": [i]})

    for s in stations:
        s["members"] = [(i, layers[i]["rPos"] - s["R0"]) for i in s["sensors"]]

    fixed_radii = []
    for i, L in enumerate(layers):
        if L["tyLay"] != 1 or L["flLay"] == 1:
            continue
        candidates = [s for s in stations
                      if L["label"] != "PIPE" and s["label"] == L["label"]
                      and np.isclose(s["L0"], half_length(L))]
        if candidates:
            s = min(candidates, key=lambda s: abs(L["rPos"] - s["R0"]))
            s["members"].append((i, L["rPos"] - s["R0"]))
        else:
            fixed_radii.append(L["rPos"])

    for s in stations:
        offsets = [dr for _, dr in s["members"]]
        s["dr_min"], s["dr_max"] = min(offsets), max(offsets)

    return stations, sorted(fixed_radii)

def build_disk_stations(layers):
    """
    Group disk layers into rigid, mirror-symmetric stations.

    Each +z measurement disk is a station. Each passive +z disk is attached to
    the nearest station with the same label and radial extent (its support).
    Every member has a -z mirror (same label and radii, z -> -z) that moves
    with it, so only +z positions are optimized.

    Returns stations sorted by z, each with 'Z0' (sensor z), 'rmin'/'rmax'
    (radial extent of the members) and 'members' as
    (layer index, mirror layer index, z offset from Z0).
    """
    def same_disk(a, b):
        return (a["label"] == b["label"] and np.isclose(a["xMin"], b["xMin"])
                and np.isclose(a["xMax"], b["xMax"]))

    def mirror_of(i):
        for j, L in enumerate(layers):
            if (L["tyLay"] == 2 and L["flLay"] == layers[i]["flLay"]
                    and same_disk(L, layers[i]) and np.isclose(L["rPos"], -layers[i]["rPos"])):
                return j
        raise ValueError(f"disk layer {i} ({layers[i]['label']} at z={layers[i]['rPos']}) has no -z mirror")

    pos = [i for i, L in enumerate(layers) if L["tyLay"] == 2 and L["rPos"] > 0]
    stations = [
        {"label": layers[i]["label"], "Z0": layers[i]["rPos"],
         "rmin": layers[i]["xMin"], "rmax": layers[i]["xMax"],
         "members": [(i, mirror_of(i), 0.0)]}
        for i in sorted((i for i in pos if layers[i]["flLay"] == 1), key=lambda i: layers[i]["rPos"])
    ]
    for i in pos:
        if layers[i]["flLay"] == 1:
            continue
        candidates = [s for s in stations if same_disk(layers[s["members"][0][0]], layers[i])]
        if not candidates:
            raise ValueError(f"passive disk {i} ({layers[i]['label']} at z={layers[i]['rPos']}) has no matching sensor disk")
        s = min(candidates, key=lambda s: abs(layers[i]["rPos"] - s["Z0"]))
        s["members"].append((i, mirror_of(i), layers[i]["rPos"] - s["Z0"]))

    for s in stations:
        offsets = [dz for _, _, dz in s["members"]]
        s["dz_min"], s["dz_max"] = min(offsets), max(offsets)
    return stations

def fixed_barrel_layers():
    """
    Barrel layers that never move (support shells), excluding the beam pipe:
    in HC the pipe runs through the whole detector by design and the disks'
    inner edge is set by the input geometry.
    """
    moved = {idx for s in STATIONS for idx, _ in s["members"]}
    return [L for i, L in enumerate(BASE_LAYERS)
            if L["tyLay"] == 1 and i not in moved and L["label"] != "PIPE"]

def disk_z_bounds(stations, gap=MIN_GAP):
    """
    |z| bounds for each disk station such that disks cannot cross.

    Two stations only constrain each other when their radial extents overlap
    (e.g. ITK and OTK disks can share a z). Neighbouring overlapping stations
    split the space at the baseline midpoint with `gap` clearance, as for the
    barrel radii. A station also stays far enough out that barrels at its
    radius can keep their minimum length, and beyond the end of every fixed
    barrel layer (support shell) it radially overlaps. Bounds are additionally limited to
    [Z_FRAC_LO, Z_FRAC_HI] x baseline and Z_ABS_MAX.
    """
    bounds = []
    for k, s in enumerate(stations):
        lo_k, hi_k = s["Z0"] + s["dz_min"], s["Z0"] + s["dz_max"]
        lo = max(Z_FRAC_LO * s["Z0"], L_MIN + CLEARANCE - s["dz_min"])
        hi = min(Z_FRAC_HI * s["Z0"], Z_ABS_MAX - s["dz_max"])
        # Fixed barrel layers (support shells) that the disk radially overlaps
        # cannot be shortened, so the disk must stay beyond their end
        for b in fixed_barrel_layers():
            if s["rmin"] - CLEARANCE < b["rPos"] < s["rmax"] + CLEARANCE:
                lo = max(lo, max(abs(b["xMin"]), abs(b["xMax"])) + CLEARANCE - s["dz_min"])
        for n in stations:
            if n is s or not (n["rmin"] - gap < s["rmax"] and s["rmin"] < n["rmax"] + gap):
                continue
            lo_n, hi_n = n["Z0"] + n["dz_min"], n["Z0"] + n["dz_max"]
            if hi_n <= lo_k:      # n is below s
                lo = max(lo, 0.5 * (hi_n + lo_k) + 0.5 * gap - s["dz_min"])
            elif lo_n >= hi_k:    # n is above s
                hi = min(hi, 0.5 * (hi_k + lo_n) - 0.5 * gap - s["dz_max"])
        if not lo < hi:
            raise ValueError(f"Disk station {k} ({s['label']} at z={s['Z0']:.4f}) has empty z range [{lo:.4f}, {hi:.4f}]")
        if not lo <= s["Z0"] <= hi:
            print(f"[WARNING] baseline z={s['Z0']:.4f} of disk station {k} outside its bounds [{lo:.4f}, {hi:.4f}]")
        bounds.append((lo, hi))
    return bounds

def disk_rmin(rmin0, z0, z):
    """
    Inner radius of a disk moved from |z0| to |z|: keeps the baseline margin to
    the nozzle when moving outwards and never shrinks below the baseline.
    Without a nozzle (HC) the inner radius is unchanged.
    """
    if not has_nozzle():
        return rmin0
    return max(rmin0, rmin0 + float(nozzle_profile(abs(z))) - float(nozzle_profile(abs(z0))))

def interface_radii(stations, fixed_radii):
    """Fixed barrel radii at the inner/outer tracker interface (see INTERFACE_LABELS)."""
    inner = [s["R0"] + s["dr_max"] for s in stations if s["label"] == INTERFACE_LABELS[0]]
    outer = [s["R0"] + s["dr_min"] for s in stations if s["label"] == INTERFACE_LABELS[1]]
    if not inner or not outer:
        return set()
    return {r for r in fixed_radii if max(inner) < r < min(outer)}

def station_radius_bounds(stations, free, fixed_radii, gap=MIN_GAP):
    """
    Radius bounds for each free station such that no two layers can cross.

    Every station keeps at least `gap` from fixed layers (beam pipe, support
    shells) and from stations that are not optimized, and INTERFACE_GAP from
    the fixed layers at the inner/outer tracker interface. Two neighbouring free
    stations split the space between them at the baseline midpoint, so any
    point in the box is a valid ordering with at least `gap` clearance.
    Bounds are additionally limited to [R_FRAC_LO, R_FRAC_HI] x baseline.
    """
    # Everything a free station can bump into, as (inner, outer, is_free_station)
    items = [(r, r, False) for r in fixed_radii]
    for k, s in enumerate(stations):
        items.append((s["R0"] + s["dr_min"], s["R0"] + s["dr_max"], k in free))
    items.sort()
    interface = interface_radii(stations, fixed_radii)

    def clearance(item):
        """Gap to keep from a fixed neighbour: wider at the inner/outer tracker interface."""
        return INTERFACE_GAP if item[0] in interface else gap

    bounds = []
    for k in free:
        s = stations[k]
        inner, outer = s["R0"] + s["dr_min"], s["R0"] + s["dr_max"]
        pos = items.index((inner, outer, True))

        lo = R_FRAC_LO * s["R0"]
        if pos > 0:
            p_in, p_out, p_free = items[pos - 1]
            edge = 0.5 * (p_out + inner) + 0.5 * gap if p_free else p_out + clearance(items[pos - 1])
            lo = max(lo, edge - s["dr_min"])

        hi = min(R_FRAC_HI * s["R0"], R_ABS_MAX - s["dr_max"])
        if pos < len(items) - 1:
            n_in, n_out, n_free = items[pos + 1]
            edge = 0.5 * (outer + n_in) - 0.5 * gap if n_free else n_in - clearance(items[pos + 1])
            hi = min(hi, edge - s["dr_max"])

        if not lo < hi:
            raise ValueError(f"Station {k} ({s['label']} at R={s['R0']:.4f}) has empty radius range [{lo:.4f}, {hi:.4f}]")
        if not lo <= s["R0"] <= hi:
            print(f"[WARNING] baseline R={s['R0']:.4f} of station {k} outside its bounds [{lo:.4f}, {hi:.4f}]")
        bounds.append((lo, hi))
    return bounds

def write_geo_from_theta(theta, out_path=GEO_OPT):
    """
    theta: [R_1..R_N, f_1..f_N, Z_1..Z_M] for the free stations and disks.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape[0] != THETA_DIM:
        raise ValueError(f"theta has length {theta.shape[0]}, expected {THETA_DIM}")

    write_layers(build_layers_from_theta(theta), out_path)

def write_layers(layers, out_path):
    """Write a list of layer dicts in the GeoCLD.txt format read by SolGeom::GeoRead."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for L in layers:
            f.write(
                f"{L['tyLay']} {L['label']} "
                f"{L['xMin']:.6g} {L['xMax']:.6g} {L['rPos']:.6g} "
                f"{L['thLay']:.6g} {L['rlLay']:.6g} {L['nmLay']:d} "
                f"{L['stLayU']:.6g} {L['stLayL']:.6g} "
                f"{L['sgLayU']:.6g} {L['sgLayL']:.6g} {L['flLay']:d}\n"
            )

def baseline_theta():
    """Parameter vector reproducing the base geometry."""
    thetas_r = [s["R0"] for s in FREE_STATIONS]
    thetas_f = []
    for s in FREE_STATIONS:
        L_max = station_length_max(s, s["R0"], BASE_LAYERS)
        if s["L0"] > L_max + 1e-9:
            print(f"[WARNING] baseline {s['label']} station at R={s['R0']:.4f} has L={s['L0']:.4f} > allowed {L_max:.4f}")
        span = L_max - L_MIN
        thetas_f.append(float(np.clip((s["L0"] - L_MIN) / span, 0.0, 1.0)) if span > 0 else 0.0)
    thetas_z = [s["Z0"] for s in FREE_DISK_STATIONS]
    return np.array(thetas_r + thetas_f + thetas_z, dtype=float)

def checkpoint_file():
    return OPT_DIR / f"bo_checkpoint_{COLLIDER}.json"

def best_geometry_file():
    """Where a run writes its best geometry (and where --resume falls back to)."""
    return OPT_DIR / f"GeoBEST_{COLLIDER}.txt"

def checkpoint_config():
    """What must match for a checkpoint to be resumable."""
    return {
        "collider": COLLIDER,
        "theta_dim": THETA_DIM,
        "stations": [[s["label"], round(s["R0"], 6)] for s in FREE_STATIONS],
        "disk_stations": [[s["label"], round(s["Z0"], 6)] for s in FREE_DISK_STATIONS],
        "track_grid": [N_THETA, THETA_MIN_DEG, N_PT, PT_MIN, PT_MAX],
        "loss": {"version": 2, "n_hits_min": N_HITS_MIN, "seed_hits_min": SEED_HITS_MIN,
                 "seed_labels": list(SEED_LABELS), "hit_eff": HIT_EFF,
                 "hit_eff_default": HIT_EFF_DEFAULT, "eff_weight": EFF_WEIGHT},
    }

def save_checkpoint(x_iters, func_vals, path=None):
    """Write all evaluations so far; atomic, so an interrupted run never leaves a broken file."""
    path = Path(path or checkpoint_file())
    data = {
        "config": checkpoint_config(),
        "x": [[float(v) for v in x] for x in x_iters],
        "y": [float(v) for v in func_vals],
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)

def load_checkpoint(path=None):
    """Return (X, y) from a checkpoint written by the same configuration."""
    path = Path(path or checkpoint_file())
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint to resume from: {path}")
    with open(path) as f:
        data = json.load(f)
    expected = checkpoint_config()
    if data["config"] != expected:
        diff = {k: (data["config"].get(k), v) for k, v in expected.items() if data["config"].get(k) != v}
        raise ValueError(f"checkpoint {path} was written with a different configuration "
                         f"(saved vs current): {diff}")
    return data["x"], data["y"]

def theta_from_geometry(geo_path, tol=1e-5):
    """
    Parameter vector reproducing a geometry file written by this optimizer
    (e.g. an earlier GeoBEST.txt) in the current configuration.

    Radii and disk |z| are read from the stations' reference layers, length
    fractions are recomputed against the disk positions in that file. Raises
    ValueError if the file has a different layer structure or cannot be
    reproduced (e.g. layers that are fixed now were moved there).
    """
    layers = load_layers(geo_path)
    if [(L["tyLay"], L["label"], L["flLay"]) for L in layers] != \
       [(L["tyLay"], L["label"], L["flLay"]) for L in BASE_LAYERS]:
        raise ValueError(f"{geo_path}: layer structure differs from the base geometry "
                         f"in collider mode {COLLIDER}")
    R = [layers[s["sensors"][0]]["rPos"] for s in FREE_STATIONS]
    Z = [layers[s["members"][0][0]]["rPos"] for s in FREE_DISK_STATIONS]
    placed = build_layers_from_theta(np.r_[R, np.zeros(N_FREE_BARREL), Z])
    F = []
    for s, r in zip(FREE_STATIONS, R):
        span = station_length_max(s, r, placed) - L_MIN
        L = layers[s["sensors"][0]]["xMax"]
        F.append(float(np.clip((L - L_MIN) / span, 0.0, 1.0)) if span > 0 else 0.0)
    theta = np.r_[R, F, Z]
    dev = max(abs(a[k] - b[k]) for a, b in zip(build_layers_from_theta(theta), layers)
              for k in ("rPos", "xMin", "xMax"))
    if dev > tol:
        raise ValueError(f"{geo_path}: cannot be reproduced in the current configuration "
                         f"(max deviation {dev:.2e} m)")
    return theta

def add_known_geometry(geo, space, X, Y):
    """
    Append the point reproducing geometry file `geo` to (X, Y), evaluating it
    unless it is already known. Returns False (with a warning) if the file
    cannot be used in the current configuration or search space.
    """
    try:
        th = theta_from_geometry(geo)
    except ValueError as e:
        print(f"[WARNING] {e}")
        return False
    # Values read back from a 6-digit text file can sit a hair outside a bound
    th = [float(np.clip(v, d.low, d.high)) if d.low - 1e-6 <= v <= d.high + 1e-6 else float(v)
          for d, v in zip(space, th)]
    if not all(d.low <= v <= d.high for d, v in zip(space, th)):
        print(f"[WARNING] {geo} lies outside the current search space")
        return False
    key = tuple(np.round(th, 12))
    if any(tuple(np.round(x, 12)) == key for x in X):
        print(f"{geo} already known: loss = {_cache[key]:.4f}")
        return True
    X.append(th)
    Y.append(score(th))
    print(f"{geo} added as a known point: loss = {Y[-1]:.4f}")
    return True

def build_search_space():
    """Search space of the current configuration (prints the ranges)."""
    space = []

    # Radii for free stations: ±30% around baseline, clipped so layers cannot
    # cross each other, the beam pipe or fixed support shells
    free = [STATIONS.index(s) for s in FREE_STATIONS]
    r_bounds = station_radius_bounds(STATIONS, free, FIXED_RADII)
    for i, (r_min, r_max) in enumerate(r_bounds):
        space.append(Real(r_min, r_max, name=f"R_barrel_{i}"))

    # Half-lengths as a fraction of the longest nozzle/disk-safe length
    for i in range(N_FREE_BARREL):
        space.append(Real(0.0, 1.0, name=f"f_barrel_{i}"))

    # |z| of free disk stations (mirrored to -z), ordered and non-overlapping
    z_bounds = disk_z_bounds(DISK_STATIONS)
    z_bounds = [b for s, b in zip(DISK_STATIONS, z_bounds) if s in FREE_DISK_STATIONS]
    for i, (z_min, z_max) in enumerate(z_bounds):
        space.append(Real(z_min, z_max, name=f"Z_disk_{i}"))

    print(f"Search space defined with {len(space)} dimensions")
    print(f"  Radii (ordered, non-overlapping):")
    for s, (r_min, r_max) in zip(FREE_STATIONS, r_bounds):
        print(f"    {s['label']:<4} R0={s['R0']:.4f}  ->  [{r_min:.4f}, {r_max:.4f}] m")
    print(f"  Lengths: L = {L_MIN:.3f} + f * (L_max(R) - {L_MIN:.3f}), f in [0, 1], L_max <= {L_ABS_MAX:.3f} m")
    if N_FREE_DISK > 0:
        print(f"  Disk |z| (mirrored to -z, ordered, non-overlapping):")
        for s, (z_min, z_max) in zip(FREE_DISK_STATIONS, z_bounds):
            print(f"    {s['label']:<6} Z0={s['Z0']:.4f}  ->  [{z_min:.4f}, {z_max:.4f}] m")
    print()
    return space

def collect_known_points(space, n_initial_points, resume, seed_files):
    """
    Points the optimizer starts from: baseline, checkpoint (resume), previous
    best geometry (resume without checkpoint) and seed files.
    Returns (X, Y, y_base, n_random, seed).
    """
    x_base = [float(v) for v in baseline_theta()]

    def in_space(x):
        return all(d.low - 1e-12 <= v <= d.high + 1e-12 for d, v in zip(space, x))

    # The baseline is always the reference (loss 1.0), but with tighter rules
    # (e.g. INTERFACE_GAP) it can lie outside the search space and then cannot
    # be given to the optimizer as a known point.
    base_ok = in_space(x_base)
    if not base_ok:
        print("[NOTE] the baseline geometry is outside the current search space: "
              "it stays the reference (loss 1.0) but is not a starting point")

    if resume and not checkpoint_file().exists():
        # No checkpoint: fall back to the previous run's best geometry
        y_base = score(x_base)
        X, Y = ([x_base], [y_base]) if base_ok else ([], [])
        candidates = [best_geometry_file(), OPT_DIR / "GeoBEST.txt"]
        print(f"No checkpoint {checkpoint_file()}: resuming from the previous best geometry")
        if not any(f.exists() and add_known_geometry(f, space, X, Y) for f in candidates):
            raise FileNotFoundError(f"nothing to resume from: no {checkpoint_file().name} "
                                    f"and no usable {' / '.join(f.name for f in candidates)} in {OPT_DIR}")
        n_random = n_initial_points   # the GP only knows these points: explore again
        seed = 42 + len(X)
        print()
    elif resume:
        X, Y = load_checkpoint()
        # Points outside the current search space (bounds changed since) cannot be used
        inside = [in_space(x) for x in X]
        n_out = inside.count(False)
        X = [x for x, ok in zip(X, inside) if ok]
        Y = [y for y, ok in zip(Y, inside) if ok]
        print(f"Resuming from {checkpoint_file()}: {len(X)} evaluations loaded"
              + (f", {n_out} outside the current bounds dropped" if n_out else ""))
        if not X:
            raise RuntimeError("no usable evaluations in the checkpoint")
        # Known points are never re-evaluated
        for x, y in zip(X, Y):
            _cache[tuple(np.round(x, 12))] = y
        if tuple(np.round(x_base, 12)) not in _cache:
            y_base = score(x_base)
            if base_ok:
                X.append(x_base)
                Y.append(y_base)
        y_base = _cache[tuple(np.round(x_base, 12))]
        n_random = max(0, n_initial_points - (len(X) - int(base_ok)))
        seed = 42 + len(X)          # do not replay the random points of the first run
        print(f"Best so far: {min(Y):.4f}; {n_random} random points left, then GP\n")
    else:
        if checkpoint_file().exists():
            prev = checkpoint_file().with_suffix(".prev.json")
            os.replace(checkpoint_file(), prev)
            print(f"Previous checkpoint moved to {prev}")
        y_base = score(x_base)
        X, Y = ([x_base], [y_base]) if base_ok else ([], [])
        n_random = n_initial_points
        seed = 42
        print(f"Baseline loss = {y_base:.4f}" + (" (seeded into the optimizer)" if base_ok else "") + "\n")

    # Extra known points from geometry files (e.g. the best result of an earlier run)
    for geo in seed_files:
        add_known_geometry(geo, space, X, Y)
    if seed_files:
        print()

    # The optimizer needs at least one point to start from
    if not X and n_random < 1:
        n_random = 1
    return X, Y, y_base, n_random, seed

def evaluate_batch(thetas, workers):
    """
    Evaluate several parameter vectors in parallel, at most `workers` ROOT jobs
    at a time, each in its own directory opt/workers/wNN (geometry, metrics,
    root.log). Returns the losses in input order.
    """
    slots = Queue()
    for i in range(workers):
        d = OPT_DIR / "workers" / f"w{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        slots.put(d)

    def run(theta):
        d = slots.get()
        try:
            return score(theta, workdir=d)
        finally:
            slots.put(d)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, thetas))

def run_skopt(space, X, Y, n_calls, n_random, seed, workers):
    """Sequential GP-EI with scikit-optimize (one evaluation per GP fit)."""
    # skopt needs n_calls >= random points: use only as many as the budget allows
    if n_random > n_calls:
        print(f"[NOTE] {n_random} random initial points requested but only {n_calls} calls: "
              f"using {n_calls} random points, the GP takes over in a later --resume")
        n_random = n_calls

    def checkpoint(res):
        save_checkpoint(res.x_iters, res.func_vals)

    with threadpool_limits(limits=workers):
        res = gp_minimize(
            func=score,
            dimensions=space,
            x0=X or None,            # skopt rejects an empty list
            y0=Y or None,
            acq_func="EI",           # Expected Improvement
            n_calls=n_calls,
            n_initial_points=n_random,
            noise=GP_NOISE,
            random_state=seed,
            callback=[checkpoint],
            verbose=True,
        )
    return [list(map(float, x)) for x in res.x_iters], [float(y) for y in res.func_vals]

def run_botorch(space, X, Y, n_calls, n_random, seed, batch, workers, acq="ts"):
    """
    Batch Bayesian optimization with BoTorch: each round fits one GP to all
    known points and proposes `batch` geometries, which are then evaluated in
    parallel (`workers` ROOT jobs at a time).

    The GP is fitted to -log(loss): a few very bad geometries (loss >> 1)
    otherwise dominate the outcome scaling and wash out the differences near
    the optimum (hold-out rank correlation 0.95 vs 0.60 on real data).

    acq: "ts"     Thompson sampling in a trust region around the best point
                  (TuRBO, default): one GP posterior draw per batch slot; ~1 s
                  per batch. Plain global Thompson sampling barely improves in
                  35 dimensions, the trust region focuses it.
         "qlogei" q-LogEI optimized point by point; better per point but its
                  cost grows quickly with batch size and number of points.
    """
    import torch
    from torch.quasirandom import SobolEngine
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.optim import optimize_acqf
    from botorch.generation import MaxPosteriorSampling
    from gpytorch.mlls import ExactMarginalLogLikelihood

    torch.set_num_threads(workers)
    dtype = torch.double
    lo = np.array([d.low for d in space])
    hi = np.array([d.high for d in space])
    dim = len(space)
    unit_bounds = torch.stack([torch.zeros(dim, dtype=dtype), torch.ones(dim, dtype=dtype)])
    sobol = SobolEngine(dim, scramble=True, seed=seed)

    X, Y = list(X), list(Y)
    done, rnd = 0, 0
    # Trust-region state (TuRBO)
    tr_len = TR_LENGTH_INIT
    tr_fail_rounds = max(1, int(np.ceil(max(4.0 / batch, dim / batch))))
    n_succ = n_fail = 0
    while done < n_calls:
        q = min(batch, n_calls - done)
        t0 = time.time()
        if n_random > 0 or len(X) < 2:
            # Initial exploration: quasi-random (Sobol) points
            q = min(q, max(n_random, 1))
            cand = sobol.draw(q).to(dtype)
            n_random -= q
            how = "random"
        else:
            train_X = torch.tensor((np.array(X) - lo) / (hi - lo), dtype=dtype)
            # BoTorch maximizes: fit -log(loss), which tames the bad-geometry outliers
            train_Y = -torch.log(torch.tensor(Y, dtype=dtype)).unsqueeze(-1)
            # Deterministic objective: tiny noise, relative to the spread (avoids gpytorch's 1e-6 floor)
            yvar = torch.full_like(train_Y, GP_NOISE_REL * max(train_Y.var().item(), 1e-12))
            model = SingleTaskGP(train_X, train_Y, train_Yvar=yvar, outcome_transform=Standardize(m=1))
            with threadpool_limits(limits=workers):
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
                if acq == "qlogei":
                    cand, _ = optimize_acqf(qLogExpectedImprovement(model, best_f=train_Y.max()),
                                            bounds=unit_bounds, q=q, num_restarts=10,
                                            raw_samples=512, sequential=True)
                    how = "GP q-LogEI"
                else:
                    torch.manual_seed(seed + rnd)
                    x_best = train_X[train_Y.argmax()]
                    # Trust region: box around the best point, wider along parameters
                    # the GP finds less sensitive (longer length scales)
                    ls = model.covar_module.lengthscale.detach().squeeze()
                    w = ls / ls.mean()
                    w = w / torch.prod(w.pow(1.0 / dim))
                    tr_lb = (x_best - w * tr_len / 2).clamp(0, 1)
                    tr_ub = (x_best + w * tr_len / 2).clamp(0, 1)
                    # Candidates: perturb a random subset of the best point's coordinates
                    pert = tr_lb + (tr_ub - tr_lb) * SobolEngine(dim, scramble=True, seed=seed + rnd).draw(TS_CANDIDATES).to(dtype)
                    mask = torch.rand(TS_CANDIDATES, dim) <= min(1.0, 20.0 / dim)
                    mask[torch.arange(TS_CANDIDATES), torch.randint(dim, (TS_CANDIDATES,))] = True
                    cands = torch.where(mask, pert, x_best.expand(TS_CANDIDATES, dim))
                    with torch.no_grad():
                        cand = MaxPosteriorSampling(model=model, replacement=False)(cands, num_samples=q)
                    how = f"GP Thompson (trust region {tr_len:.3f})"
        t_prop = time.time() - t0

        thetas = [list(map(float, lo + np.clip(c, 0.0, 1.0) * (hi - lo))) for c in cand.detach().numpy()]
        t0 = time.time()
        ys = evaluate_batch(thetas, workers)
        t_eval = time.time() - t0

        best_before = min(Y) if Y else np.inf
        X += thetas
        Y += [float(y) for y in ys]
        done += len(thetas)
        rnd += 1

        # Trust-region update (only once the GP is driving the search)
        if how.startswith("GP Thompson"):
            if min(ys) < best_before - TR_IMPROVEMENT * abs(best_before):
                n_succ, n_fail = n_succ + 1, 0
            else:
                n_succ, n_fail = 0, n_fail + 1
            if n_succ >= TR_SUCCESS_ROUNDS:
                tr_len, n_succ = min(2.0 * tr_len, TR_LENGTH_MAX), 0
            elif n_fail >= tr_fail_rounds:
                tr_len, n_fail = tr_len / 2.0, 0
            if tr_len < TR_LENGTH_MIN:
                print(f"[NOTE] trust region collapsed: restarting it at {TR_LENGTH_INIT}")
                tr_len = TR_LENGTH_INIT
        save_checkpoint(X, Y)
        print(f"[round {rnd}] {len(thetas)} {how} points: batch best {min(ys):.4f}, "
              f"overall best {min(Y):.4f} ({len(Y)} points) | propose {t_prop:.1f} s, "
              f"evaluate {t_eval:.1f} s | {done}/{n_calls} calls")
    return X, Y

def run_bayes_optimization(n_calls=50, n_initial_points=20, resume=False, seed_files=(),
                           backend="botorch", batch_size=None, workers=None, acq="ts"):
    """
    Run Bayesian optimization to find best layer configuration.

    The baseline geometry is evaluated and passed to the optimizer as a
    starting point (if it lies in the search space), so the reported optimum
    can never be worse than it.

    Every evaluation is saved to checkpoint_file() as it happens. With
    resume=True, the evaluations in that file are given to the optimizer as
    already-known points (no ROOT re-runs) and the search continues from them.
    If there is no checkpoint (e.g. a run made before checkpointing existed),
    resume starts from the best geometry of the previous run instead
    (best_geometry_file(), or the older name opt/GeoBEST.txt) plus the baseline.

    Args:
        n_calls: Number of new evaluations (baseline and resumed points excluded)
        n_initial_points: Number of random evaluations before the GP takes
            over, counted over the whole run including resumed ones
        resume: Continue from checkpoint_file() instead of starting fresh
        seed_files: Geometry files (e.g. an earlier GeoBEST.txt) added as known
            points; each costs one evaluation unless already in the checkpoint
        backend: "botorch" (batches evaluated in parallel) or "skopt" (sequential)
        batch_size: geometries proposed per GP fit (botorch); default = workers
        workers: parallel ROOT jobs and GP threads; default cpu_cap()
        acq: batch selection for botorch, "ts" (Thompson sampling) or "qlogei"
    """
    workers = workers or cpu_cap()
    batch_size = batch_size or workers

    print(f"\n{'='*60}")
    print(f"Starting Bayesian Optimization")
    print(f"  Free barrel stations: {N_FREE_BARREL}")
    print(f"  Free disk layers:     {N_FREE_DISK}")
    print(f"  Total parameters:     {THETA_DIM}")
    print(f"  Optimization calls:   {n_calls}")
    print(f"  Tracks per call:      {N_THETA} theta x {N_PT} pT = {N_THETA * N_PT}")
    print(f"  Backend:              {backend}" + (f" (batch {batch_size}, {acq})" if backend == "botorch" else ""))
    print(f"  CPU cores used:       {workers} (cap {CPU_FRACTION:.0%} of the node)")
    print(f"{'='*60}\n")

    space = build_search_space()
    X, Y, y_base, n_random, seed = collect_known_points(space, n_initial_points, resume, seed_files)

    if backend == "skopt":
        X, Y = run_skopt(space, X, Y, n_calls, n_random, seed, workers)
    elif backend == "botorch":
        X, Y = run_botorch(space, X, Y, n_calls, n_random, seed, batch_size, workers, acq=acq)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    ibest = int(np.argmin(Y))
    res = SimpleNamespace(x=X[ibest], fun=Y[ibest], x_iters=X, func_vals=np.array(Y))
    y0 = y_base

    best_layers = build_layers_from_theta(res.x)
    best_L = [best_layers[s["sensors"][0]]["xMax"] for s in FREE_STATIONS]

    print(f"\n{'='*60}")
    print("=== Optimization Finished ===")
    print(f"{'='*60}")
    print(f"Best loss:   {res.fun:.6f}  (baseline = {y0:.6f})")
    print(f"Best params: {res.x}")
    print(f"\nBest radii (m):        {res.x[:N_FREE_BARREL]}")
    print(f"Best half-lengths (m): {best_L}")
    if N_FREE_DISK > 0:
        print(f"Best disk z (m):      {res.x[2*N_FREE_BARREL:]}")

    # Write best geometry file
    best_file = best_geometry_file()
    write_geo_from_theta(res.x, out_path=best_file)
    print(f"\nBest geometry written to: {best_file}")
    draw_from_file(best_file)
    print(f"Geometry plot saved to: {OPT_DIR / 'geometry.png'}")
    print(f"All {len(res.func_vals)} evaluations saved in: {checkpoint_file()}")

    return res

def build_layers_from_theta(theta):
    '''
    Given θ = [R_1..R_N, f_1..f_N, Z_1..Z_M], return a fresh list of layer
    dicts. Disk stations are placed first (mirrored to -z, inner radius
    following the nozzle, see disk_rmin); then each free barrel station moves rigidly to
    radius R (members keep their radial offsets) and gets half-length
    L = L_MIN + f * (L_max - L_MIN), where L_max is the longest length that
    clears the nozzle and every disk at that radius.
    '''
    theta = np.asarray(theta, dtype=float)
    if theta.shape[0] != THETA_DIM:
        raise ValueError(f"theta length {theta.shape[0]} != 2 * N_FREE_BARREL + N_FREE_DISK ({THETA_DIM})")

    R = theta[:N_FREE_BARREL]
    F = theta[N_FREE_BARREL:2*N_FREE_BARREL]
    Z = theta[2*N_FREE_BARREL:]

    # copy baseline layers
    layers = [dict(L0) for L0 in BASE_LAYERS]

    # Update disk stations: members keep their z offset, -z mirrors follow
    for z, s in zip(Z, FREE_DISK_STATIONS):
        for idx, mirror, dz in s["members"]:
            z_new = float(z + dz)
            for j, sign in ((idx, 1.0), (mirror, -1.0)):
                layers[j]["rPos"] = sign * z_new
                layers[j]["xMin"] = disk_rmin(BASE_LAYERS[j]["xMin"], BASE_LAYERS[j]["rPos"], z_new)

    # Update barrel stations
    for r, f, s in zip(R, F, FREE_STATIONS):
        ell = length_from_fraction(f, station_length_max(s, r, layers))
        for idx, dr in s["members"]:
            layers[idx]["rPos"] = float(r + dr)
            # enforce symmetric barrel: xMin = -L, xMax = +L
            layers[idx]["xMin"] = float(-ell)
            layers[idx]["xMax"] = float(+ell)

    return layers

def tracker_envelope(layers):
    """
    Bounding box (r_max, |z|_max) of all layers except the beam pipe: barrels
    contribute their radius and half-length, disks their outer radius and |z|.
    """
    r_max = z_max = 0.0
    for L in layers:
        if L["label"] == "PIPE":
            continue
        if L["tyLay"] == 1:
            r, z = L["rPos"], max(abs(L["xMin"]), abs(L["xMax"]))
        else:
            r, z = L["xMax"], abs(L["rPos"])
        r_max, z_max = max(r_max, r), max(z_max, z)
    return r_max, z_max

def initialize_optimization_config(n_stations=None, optimize_disks=False, collider="MC"):
    """
    Initialize global configuration for optimization.

    Args:
        n_stations: Number of innermost barrel stations to optimize
                    (default: all). VTX doublets count as one station.
        optimize_disks: Whether to also optimize disk layers (default: False)
        collider: "MC" (muon collider: nozzle, no beam pipe layer) or
                  "HC" (hadron collider: beam pipe, no nozzle)
    """
    global R_ABS_MAX, L_ABS_MAX, Z_ABS_MAX
    global COLLIDER, _baseline, BASE_LAYERS, STATIONS, FIXED_RADII, FREE_STATIONS, N_FREE_BARREL
    global FREE_BARREL_IDX, DISK_STATIONS, FREE_DISK_STATIONS, N_FREE_DISK, THETA_DIM

    if collider not in COLLIDERS:
        raise ValueError(f"collider must be one of {COLLIDERS}, got {collider!r}")
    COLLIDER = collider
    _baseline = None
    _cache.clear()

    # Load base geometry
    BASE_LAYERS = load_layers(GEO_BASE)
    print(f"Loaded {len(BASE_LAYERS)} layers from {GEO_BASE}")
    # Hard outer envelope = bounding box of the base geometry (beam pipe excluded)
    R_ABS_MAX, Z_env = tracker_envelope(BASE_LAYERS)
    L_ABS_MAX = Z_ABS_MAX = Z_env
    print(f"Envelope (from base geometry): r <= {R_ABS_MAX:.4f} m, |z| <= {Z_env:.4f} m")

    if has_nozzle():
        n_pipe = sum(L["label"] == "PIPE" for L in BASE_LAYERS)
        BASE_LAYERS = [L for L in BASE_LAYERS if L["label"] != "PIPE"]
        print(f"Collider MC: nozzle on, {n_pipe} beam pipe layer(s) removed")
    else:
        print("Collider HC: no nozzle, beam pipe kept")

    STATIONS, FIXED_RADII = build_stations(BASE_LAYERS)
    FREE_STATIONS = STATIONS if n_stations is None else STATIONS[:n_stations]
    N_FREE_BARREL = len(FREE_STATIONS)
    FREE_BARREL_IDX = [idx for s in FREE_STATIONS for idx, _ in s["members"]]

    print(f"\nBarrel stations found: {len(STATIONS)}")
    print(f"Optimizing {N_FREE_BARREL} stations:")
    for s in FREE_STATIONS:
        sensors  = ", ".join(f"{BASE_LAYERS[i]['rPos']:.4f}" for i in s["sensors"])
        supports = ", ".join(f"{BASE_LAYERS[i]['rPos']:.4f}" for i, _ in s["members"] if i not in s["sensors"])
        L_max = station_length_max(s, s["R0"], BASE_LAYERS)
        print(f"  {s['label']:<4} sensors R=[{sensors}]  supports R=[{supports}]  L={s['L0']:.4f} m (max {L_max:.4f})")
    print(f"Fixed barrel layers at R = {[round(r, 4) for r in FIXED_RADII]}")

    # Optionally optimize disk layers
    DISK_STATIONS = build_disk_stations(BASE_LAYERS)
    if optimize_disks:
        FREE_DISK_STATIONS = DISK_STATIONS
        N_FREE_DISK = len(FREE_DISK_STATIONS)
        print(f"\nOptimizing {N_FREE_DISK} disk stations (each mirrored to -z):")
        for s in FREE_DISK_STATIONS:
            passive = [f"{BASE_LAYERS[i]['rPos']:.4f}" for i, _, dz in s["members"] if dz != 0.0]
            print(f"  {s['label']:<6} z={s['Z0']:.4f}  r=[{s['rmin']:.4f}, {s['rmax']:.4f}]  supports z={passive}")
    else:
        FREE_DISK_STATIONS = []
        N_FREE_DISK = 0
        print("\nKeeping all disk layers FIXED")

    THETA_DIM = 2 * N_FREE_BARREL + N_FREE_DISK
    print(f"\nTotal optimization parameters: {THETA_DIM}")
    print(f"  {N_FREE_BARREL} radii + {N_FREE_BARREL} length fractions + {N_FREE_DISK} disk |z| positions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian optimization of the tracker layout")
    parser.add_argument("--n-calls", type=int, default=200,
                        help="optimizer evaluations after the baseline (default: 200)")
    parser.add_argument("--n-initial", type=int, default=40,
                        help="random evaluations before the GP takes over (default: 40)")
    parser.add_argument("--n-stations", type=int, default=None,
                        help="optimize only the N innermost barrel stations (default: all)")
    parser.add_argument("--no-disks", action="store_true",
                        help="keep the disks fixed (default: disks are optimized too)")
    parser.add_argument("--resume", action="store_true",
                        help="continue from opt/bo_checkpoint_<collider>.json (same configuration), "
                             "or, if there is none, from the previous best geometry "
                             "opt/GeoBEST_<collider>.txt; --n-calls then counts the additional evaluations")
    parser.add_argument("--seed", nargs="+", default=[], metavar="GEO",
                        help="geometry file(s) written by this optimizer (e.g. an earlier "
                             "opt/GeoBEST.txt) to add as known starting points")
    parser.add_argument("--backend", choices=("botorch", "skopt"), default="botorch",
                        help="botorch: batches of geometries evaluated in parallel (default); "
                             "skopt: one evaluation at a time")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"parallel ROOT jobs and GP threads (default: {CPU_FRACTION * 100:.0f}%% "
                             f"of the cores = {cpu_cap()})")
    parser.add_argument("--acq", choices=("ts", "qlogei"), default="ts",
                        help="botorch batch selection: ts = Thompson sampling (fast, default), "
                             "qlogei = q-LogEI (slow for large batches)")
    parser.add_argument("--batch", type=int, default=None,
                        help="geometries proposed per GP fit with botorch (default: = workers)")
    parser.add_argument("--min-hits", type=int, default=N_HITS_MIN,
                        help=f"track-finding model: minimum measured hits per track (default {N_HITS_MIN})")
    parser.add_argument("--seed-hits", type=int, default=SEED_HITS_MIN,
                        help=f"track-finding model: minimum hits in the seed subsystems "
                             f"{'/'.join(SEED_LABELS)} (default {SEED_HITS_MIN})")
    parser.add_argument("--hit-eff", type=float, default=None,
                        help=f"track-finding model: hit efficiency of every layer "
                             f"(default per subsystem, HIT_EFF in optimizer.py)")
    parser.add_argument("--eff-weight", type=float, default=EFF_WEIGHT,
                        help=f"weight of the efficiency term in the loss (default {EFF_WEIGHT}; 0 = resolution only)")
    parser.add_argument("--collider", choices=COLLIDERS, default="MC",
                        help="MC: muon collider with nozzle, no beam pipe layer; "
                             "HC: hadron collider with beam pipe, no nozzle (default: MC)")
    args = parser.parse_args()
    N_HITS_MIN, SEED_HITS_MIN, EFF_WEIGHT = args.min_hits, args.seed_hits, args.eff_weight
    if args.hit_eff is not None:
        HIT_EFF = {k: args.hit_eff for k in HIT_EFF}
        HIT_EFF_DEFAULT = args.hit_eff

    # Initialize configuration: barrel stations (VTX + ITK + OTK) and disks
    initialize_optimization_config(n_stations=args.n_stations, optimize_disks=not args.no_disks,
                                   collider=args.collider)

    # Get initial parameter vector from baseline geometry
    thetas = baseline_theta()

    print(f"\nInitial parameters (baseline):")
    print(f"  Radii:            {thetas[:N_FREE_BARREL]}")
    print(f"  Length fractions: {thetas[N_FREE_BARREL:2*N_FREE_BARREL]}")
    if N_FREE_DISK > 0:
        print(f"  Disk |z|:         {thetas[2*N_FREE_BARREL:]}")

    # Write baseline geometry and draw it
    write_geo_from_theta(thetas)
    print(f"\nWrote baseline geometry to: {GEO_OPT}")
    draw_from_file(GEO_OPT, OPT_DIR / "geometry_start.png", "baseline")
    print(f"Baseline plot saved to: {OPT_DIR / 'geometry_start.png'}")

    # Run Bayesian optimization (evaluates and seeds the baseline itself)
    result = run_bayes_optimization(n_calls=args.n_calls, n_initial_points=args.n_initial,
                                    resume=args.resume, seed_files=args.seed, backend=args.backend,
                                    batch_size=args.batch, workers=args.workers, acq=args.acq)