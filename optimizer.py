from pathlib import Path
import subprocess
import numpy as np
import uproot
import matplotlib.pyplot as plt
from skopt.space import Real
from skopt import gp_minimize

ROOT_DIR = Path(__file__).resolve().parent          # "./"
PKG_DIR  = ROOT_DIR / "FastTrackCovariance"         # "./FastTrackCovariance"

GEO_BASE      = "/homeui/fnardi/mucoll_tracker/FastTrackCovariance/GeoCLD.txt"
GEO_OPT       = "opt/GeoOPT.txt"
METRICS_FILE  = "opt/metrics.root"
RUNMETRICS_C  = "RunMetrics.cc"
LOADALL_C     = "LoadAll.c"

_cache = {}

# ============================================================================
# GLOBAL CONFIGURATION - Initialized at module load
# ============================================================================
BASE_LAYERS = []
FREE_BARREL_IDX = []
N_FREE_BARREL = 0
FREE_DISK_IDX = []
N_FREE_DISK = 0
THETA_DIM = 0

def compute_loss_from_metrics(metrics_path=METRICS_FILE):
    """
    Compute tracking performance loss from ROOT metrics file.

    Returns a weighted combination of:
      - Relative pT resolution: σ(pT)/pT
      - Impact parameter resolution: σ(d0) in mm

    Lower is better.
    """
    with uproot.open(metrics_path) as f:
        tree = f["metrics"]
        arr = tree.arrays(library="np")

    spt_rel = arr["spt_rel"]    # σ(pT)/pT (dimensionless)
    sd0_um  = arr["sd0_um"]     # σ(d0) in micrometers

    # Use mean tracking resolution
    L_pT = float(np.mean(spt_rel))
    L_d0 = float(np.mean(sd0_um) / 1000.0)  # convert μm to mm

    # Balanced weighting: both metrics matter
    # Typical values: L_pT ~ 0.01-0.1, L_d0 ~ 0.001-0.01 mm
    alpha, beta = 1.0, 1.0
    loss = alpha * L_pT + beta * L_d0

    return loss

def nozzle_profile(x):
    m1 = 0.1763
    m2 = 0.08474
    q2 = 9.156*0.01 # meters
    return np.where(x<1., m1*x, m2*x+q2)

def barrel_nozzle_penalty(theta, clearance=0.0, scale=0.01):
    """
    Compute penalty for barrel layers that violate nozzle clearance.

    Only checks the free barrel layers being optimized, not fixed layers
    like the beam pipe which have different constraints.
    """
    layers = build_layers_from_theta(theta)
    penalty = 0.0

    # Only check the free barrel layers being optimized
    for idx in FREE_BARREL_IDX:
        L = layers[idx]
        if L["tyLay"] != 1:
            continue

        R_b   = L["rPos"]
        z_end = max(abs(L["xMin"]), abs(L["xMax"]))  # half-length
        r_noz = float(nozzle_profile(z_end)) + clearance
        v = r_noz - R_b

        if v > 0.0:
            penalty += (v / scale) ** 2

    return penalty

def barrel_disk_overlap_penalty(theta, clearance=0.001, scale=0.01):
    """
    Compute penalty for barrel layers that overlap with disk layers.

    Checks all barrel-disk pairs and penalizes any violation of the clearance.
    """
    layers = build_layers_from_theta(theta)
    penalty = 0.0

    # Get free barrel and disk layers
    barrel_layers = [layers[i] for i in FREE_BARREL_IDX if layers[i]["tyLay"] == 1]
    disk_layers   = [layers[i] for i in FREE_DISK_IDX if layers[i]["tyLay"] == 2]

    for b in barrel_layers:
        R_b = b["rPos"]
        z_b_min = b["xMin"]
        z_b_max = b["xMax"]

        for d in disk_layers:
            z_d = d["rPos"]

            # Check if disk plane intersects barrel half-length
            if z_b_min - clearance < z_d < z_b_max + clearance:
                # Check radial overlap
                r_noz = float(nozzle_profile(abs(z_d))) + clearance
                v = r_noz - R_b
                if v > 0.0:
                    penalty += (v / scale) ** 2

    return penalty

def score(theta):
    """
    Objective function for Bayesian optimization.
    theta: [R_1, ..., R_N, L_1, ..., L_N, Z_1, ..., Z_M]
    Returns: scalar loss value (lower is better)
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
    L_b = theta[Nb:2*Nb]
    Z_d = theta[2*Nb:2*Nb + Nd] if Nd > 0 else np.array([])

    # --- basic bounds (should be enforced by optimizer, but double-check) ---
    # Absolute physical limits
    R_abs_min, R_abs_max = 0.020, 2.000  # absolute barrel radius limits
    L_abs_min, L_abs_max = 0.050, 2.500  # absolute half-length limits
    Z_abs_min, Z_abs_max = 0.01, 2.5     # disk z-range

    if (np.any(R_b < R_abs_min) or np.any(R_b > R_abs_max) or
        np.any(L_b < L_abs_min) or np.any(L_b > L_abs_max)):
        print(f"[OUT OF BOUNDS] R={R_b}, L={L_b}")
        _cache[key] = 1e6
        return 1e6

    if Nd > 0 and (np.any(Z_d < Z_abs_min) or np.any(Z_d > Z_abs_max)):
        print(f"[OUT OF BOUNDS] Z={Z_d}")
        _cache[key] = 1e6
        return 1e6

    # --- base tracking loss ---
    try:
        write_geo_from_theta(theta)
        run_root_metrics(GEO_OPT, METRICS_FILE)
        base_loss = compute_loss_from_metrics(METRICS_FILE)
    except Exception as e:
        print(f"[ERROR] Failed to compute metrics: {e}")
        _cache[key] = 1e6
        return 1e6

    # --- nozzle area forbidden ---
    nozzle_penalty = barrel_nozzle_penalty(theta, clearance=0.001, scale=0.01)
    
    # --- disk-barrel overlap forbidden ---
    overlap_penalty = barrel_disk_overlap_penalty(theta, clearance=0.001, scale=0.01)

    # Weighted loss: focus primarily on tracking performance
    loss = 5.* base_loss + 2.0 * nozzle_penalty + 2.0 * overlap_penalty

    print(
        f"R_b={[f'{r:.4f}' for r in R_b]}, L_b={[f'{l:.4f}' for l in L_b]} -> "
        f"base={base_loss:.4g}, nozzle={nozzle_penalty:.4g}, total={loss:.4g}"
    )

    _cache[key] = loss
    return loss

def run_root_metrics(geo_file=GEO_OPT, metrics_file=METRICS_FILE):
    cmd = [
        "root",
        "-b",
        "-q",
        f'{str(LOADALL_C)}("CLD")',         # compile library
        f'{str(RUNMETRICS_C)}+("{geo_file}", "{metrics_file}")',
    ]
    print("Running ROOT:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def draw_from_file(geom_file=GEO_OPT):
    # Read geometry file and draws layers accordingly
    with open(geom_file, 'r') as f:
        lines = f.readlines()
    layers = []
    for line in lines:
        parts = line.strip().split()
        layers.append([float(parts[0]),float(parts[2]),float(parts[3]),float(parts[4])])
    plt.figure(figsize=(15,10))
    for layer in layers:
        if layer[0]==1: # Barrel
            plt.plot([layer[1], layer[2]], [layer[3], layer[3]], color='blue')
        if layer[0]==2:
            plt.plot([layer[3], layer[3]], [layer[1], layer[2]], color='red')
    plt.xlim(-2.5,2.5)
    plt.savefig('opt/geometry.png')

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

def write_geo_from_theta(theta, out_path=GEO_OPT):
    """
    theta: array-like of length N_FREE, radii (m) for each free layer
           in the order of FREE_IDX.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape[0] != THETA_DIM:
        raise ValueError(f"theta has length {theta.shape[0]}, expected {THETA_DIM}")

    layers = build_layers_from_theta(theta)

    # write full geometry to file
    with open(out_path, "w") as f:
        for L in layers:
            f.write(
                f"{L['tyLay']} {L['label']} "
                f"{L['xMin']:.6g} {L['xMax']:.6g} {L['rPos']:.6g} "
                f"{L['thLay']:.6g} {L['rlLay']:.6g} {L['nmLay']:d} "
                f"{L['stLayU']:.6g} {L['stLayL']:.6g} "
                f"{L['sgLayU']:.6g} {L['sgLayL']:.6g} {L['flLay']:d}\n"
            )

def run_bayes_optimization(n_calls=50, n_initial_points=20):
    """
    Run Bayesian optimization to find best layer configuration.

    Args:
        n_calls: Total number of optimization iterations
        n_initial_points: Number of random initialization points
    """
    print(f"\n{'='*60}")
    print(f"Starting Bayesian Optimization")
    print(f"  Free barrel layers: {N_FREE_BARREL}")
    print(f"  Free disk layers:   {N_FREE_DISK}")
    print(f"  Total parameters:   {THETA_DIM}")
    print(f"  Optimization calls: {n_calls}")
    print(f"{'='*60}\n")

    # Define search space based on actual layer count
    # Get baseline radii to set reasonable ranges per layer
    baseline_radii = [BASE_LAYERS[i]["rPos"] for i in FREE_BARREL_IDX]

    # Half-lengths: 50-150 mm for inner, up to 2000 mm for outer
    L_min, L_max = 0.050, 2.000
    # Disk z-positions (if any)
    Z_min, Z_max = 0.01, 2.5

    space = []

    # Radii for free barrel layers - set ranges based on baseline values
    # Allow ±30% variation around baseline, with sensible min/max bounds
    for i in range(len(FREE_BARREL_IDX)):
        r_baseline = baseline_radii[i]
        r_min = max(0.020, r_baseline * 0.70)  # -30% but not below 20mm
        r_max = min(1.600, r_baseline * 1.30)  # +30% but not above 1600mm
        space.append(Real(r_min, r_max, name=f"R_barrel_{i}"))

    # Half-lengths for free barrel layers
    for i in range(N_FREE_BARREL):
        space.append(Real(L_min, L_max, name=f"L_barrel_{i}"))

    # Z-positions for free disk layers
    for i in range(N_FREE_DISK):
        space.append(Real(Z_min, Z_max, name=f"Z_disk_{i}"))

    print(f"Search space defined with {len(space)} dimensions")
    print(f"  Radii:   [adaptive per layer, ±30% around baseline]")
    print(f"    VTX layers: ~{baseline_radii[0]:.3f} - {baseline_radii[min(7, len(baseline_radii)-1)]:.3f} m")
    if len(baseline_radii) > 8:
        print(f"    ITK/OTK layers: ~{baseline_radii[8]:.3f} - {baseline_radii[-1]:.3f} m")
    print(f"  Lengths: [{L_min:.3f}, {L_max:.3f}] m")
    if N_FREE_DISK > 0:
        print(f"  Disk Z:  [{Z_min:.3f}, {Z_max:.3f}] m")
    print()

    res = gp_minimize(
        func=score,
        dimensions=space,
        acq_func="EI",           # Expected Improvement
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=42,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print("=== Optimization Finished ===")
    print(f"{'='*60}")
    print(f"Best loss:   {res.fun:.6f}")
    print(f"Best params: {res.x}")
    print(f"\nBest radii (m):       {res.x[:N_FREE_BARREL]}")
    print(f"Best half-lengths (m): {res.x[N_FREE_BARREL:2*N_FREE_BARREL]}")
    if N_FREE_DISK > 0:
        print(f"Best disk z (m):      {res.x[2*N_FREE_BARREL:]}")

    # Write best geometry file
    best_file = "opt/GeoBEST.txt"
    write_geo_from_theta(res.x, out_path=best_file)
    print(f"\nBest geometry written to: {best_file}")
    draw_from_file(best_file)
    print(f"Geometry plot saved to: opt/geometry.png")

    return res

def build_layers_from_theta(theta):
    '''
    Given θ = [R_1..R_N, L_1..L_N], return a fresh list of layer dicts
    with updated radii and xMin/xMax for the free barrel layers.
    '''
    theta = np.asarray(theta, dtype=float)
    if theta.shape[0] != THETA_DIM:
        raise ValueError(f"theta length {theta.shape[0]} != 2 * N_FREE_LAYERS ({THETA_DIM})")

    R = theta[:N_FREE_BARREL]
    L = theta[N_FREE_BARREL:2*N_FREE_BARREL]
    Z = theta[2*N_FREE_BARREL:]

    # copy baseline layers
    layers = [dict(L0) for L0 in BASE_LAYERS]

    # Update barrel layers
    for r, ell, idx in zip(R, L, FREE_BARREL_IDX):
        layers[idx]["rPos"] = float(r)
        # enforce symmetric barrel: xMin = -L, xMax = +L
        layers[idx]["xMin"] = float(-abs(ell))
        layers[idx]["xMax"] = float(+abs(ell))
    
    # Update disk layers
    for z, idx in zip(Z, FREE_DISK_IDX):
        layers[idx]["rPos"] = float(z)

    return layers

def initialize_optimization_config(n_inner_layers=7, optimize_disks=False):
    """
    Initialize global configuration for optimization.

    Args:
        n_inner_layers: Number of inner barrel layers to optimize (default: 7)
        optimize_disks: Whether to also optimize disk layers (default: False)
    """
    global BASE_LAYERS, FREE_BARREL_IDX, N_FREE_BARREL
    global FREE_DISK_IDX, N_FREE_DISK, THETA_DIM

    # Load base geometry
    BASE_LAYERS = load_layers(GEO_BASE)
    print(f"Loaded {len(BASE_LAYERS)} layers from {GEO_BASE}")

    # Find all barrel measurement layers (tyLay=1, flLay=1)
    all_barrel_idx = [
        i for i, L in enumerate(BASE_LAYERS)
        if L["tyLay"] == 1 and L["flLay"] == 1
    ]

    # Take only the first N inner layers
    FREE_BARREL_IDX = all_barrel_idx[:n_inner_layers]
    N_FREE_BARREL = len(FREE_BARREL_IDX)

    print(f"\nBarrel layers found: {len(all_barrel_idx)}")
    print(f"Optimizing first {N_FREE_BARREL} inner layers: {FREE_BARREL_IDX}")
    for idx in FREE_BARREL_IDX:
        L = BASE_LAYERS[idx]
        print(f"  Layer {idx}: {L['label']} at R={L['rPos']:.4f} m, "
              f"L={0.5*(L['xMax']-L['xMin']):.4f} m")

    # Optionally optimize disk layers
    if optimize_disks:
        FREE_DISK_IDX = [
            i for i, L in enumerate(BASE_LAYERS)
            if L["tyLay"] == 2 and L["flLay"] == 1
        ]
        N_FREE_DISK = len(FREE_DISK_IDX)
        print(f"\nOptimizing {N_FREE_DISK} disk layers: {FREE_DISK_IDX}")
    else:
        FREE_DISK_IDX = []
        N_FREE_DISK = 0
        print("\nKeeping all disk layers FIXED")

    THETA_DIM = 2 * N_FREE_BARREL + N_FREE_DISK
    print(f"\nTotal optimization parameters: {THETA_DIM}")
    print(f"  {N_FREE_BARREL} radii + {N_FREE_BARREL} half-lengths + {N_FREE_DISK} disk z-positions")


if __name__ == "__main__":
    # Initialize configuration: optimize all barrel layers (VTX + ITK + OTK)
    initialize_optimization_config(n_inner_layers=14, optimize_disks=False)

    # Get initial parameter vector from baseline geometry
    thetas_r = np.array([BASE_LAYERS[i]["rPos"] for i in FREE_BARREL_IDX])
    thetas_l = np.array([0.5*(BASE_LAYERS[i]["xMax"]-BASE_LAYERS[i]["xMin"]) for i in FREE_BARREL_IDX])
    thetas_z = np.array([BASE_LAYERS[i]['rPos'] for i in FREE_DISK_IDX]) if N_FREE_DISK > 0 else np.array([])

    thetas = np.concatenate([thetas_r, thetas_l, thetas_z])

    print(f"\nInitial parameters (baseline):")
    print(f"  Radii:       {thetas_r}")
    print(f"  Half-lengths: {thetas_l}")
    if N_FREE_DISK > 0:
        print(f"  Disk z:      {thetas_z}")

    # Write baseline geometry and compute initial loss
    write_geo_from_theta(thetas)
    print(f"\nWrote baseline geometry to: {GEO_OPT}")

    run_root_metrics(GEO_OPT, METRICS_FILE)
    draw_from_file(GEO_OPT)
    L0 = score(thetas)
    print(f"\n*** Baseline loss (before optimization): {L0:.6f} ***\n")

    # Run Bayesian optimization
    result = run_bayes_optimization(n_calls=50, n_initial_points=20)


