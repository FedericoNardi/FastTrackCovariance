### How to make things running

First of all, you need to load and build several libraries. The way to do it is to run the `LoadAll.c` script.
You can easily run it in a `root` bash:

```
root[0] .L LoadAll.c
root[1] LoadAll("CLD") 
```

You have to use "CLD", and thereore the geometry should always be called "GeoCLD.txt"

If you need to change the magnetic field, you have to modify the script `SolGeomCLD.cxx` (line 89), and rebuild everything using `LoadAll.c`.

Then, you can run the `CompRes.c` script to get the resolution for a given angle:

```
root[0] .L CompRes.c
root[1] CompRes(90) 
```

You will get a `.root` file with all the canvas saved (line 262 of script `CompRes.c`).

You can change the geometry without the need to build everything.

Let's have a look at the `GeoCLD.txt` file structure:

```
1 VTX -0.065 0.065 0.029 5e-05 0.0937 2 0 1.5708 5e-06 5e-06 1
1 VTX -0.065 0.065 0.031 5e-05 0.0937 2 0 1.5708 5e-06 5e-06 1
1 VTX -0.065 0.065 0.05 5e-05 0.0937 2 0 1.5708 5e-06 5e-06 1
1 VTX -0.065 0.065 0.052 5e-05 0.0937 2 0 1.5708 5e-06 5e-06 1
1 VTX -0.065 0.065 0.073 5e-05 0.0937 2 0 1.5708 5e-06 5e-06 1
1 VTX -0.065 0.065 0.075 5e-05 0.0937 2 0 1.5708 5e-06 5e-06 1
1 VTX -0.065 0.065 0.101 5e-05 0.0937 2 0 1.5708 5e-06 5e-06 1
1 VTX -0.065 0.065 0.103 5e-05 0.0937 2 0 1.5708 5e-06 5e-06 1
1 VTX -0.065 0.065 0.028 0.000557 0.0937 0 0 0 0 0 0
1 VTX -0.065 0.065 0.051 0.000557 0.0937 0 0 0 0 0 0
1 VTX -0.065 0.065 0.074 0.000557 0.0937 0 0 0 0 0 0
1 VTX -0.065 0.065 0.102 0.000557 0.0937 0 0 0 0 0 0
1 VTX -0.5 0.5 0.113 0.000337 0.0937 0 0 0 0 0 0
```

The structure is the following:

```
ftyLay = new Int_t[fNlMax];		// Layer type 1 = R (barrel) or 2 = z (forward/backward)
fLyLabl = new TString[fNlMax];	// Layer label
fxMin = new Double_t[fNlMax];	// Minimum dimension z for barrel  or R for forward
fxMax = new Double_t[fNlMax];	// Maximum dimension z for barrel  or R for forward
frPos = new Double_t[fNlMax];	// R/z location of layer
fthLay = new Double_t[fNlMax];	// Thickness (meters)
frlLay = new Double_t[fNlMax];	// Radiation length (meters)
fnmLay = new Int_t[fNlMax];		// Number of measurements in layers (1D or 2D)
fstLayU = new Double_t[fNlMax];	// Stereo angle (rad) - 0(pi/2) = axial(z) layer - Upper side
fstLayL = new Double_t[fNlMax];	// Stereo angle (rad) - 0(pi/2) = axial(z) layer - Lower side
fsgLayU = new Double_t[fNlMax];	// Resolution Upper side (meters) - 0 = no measurement
fsgLayL = new Double_t[fNlMax];	// Resolution Lower side (meters) - 0 = no measurement
fflLay = new Bool_t[fNlMax];	// measurement flag = T, scattering only = F
fEnable = new Bool_t[fNdet];	// list of enabled detectors
fDtype = new TString[fNdty];	// Array with layer labels 
fDfstLay = new Int_t[fNdty];	// Array with start layer
```
---

## Tracker layout optimization (`optimizer.py`)

`optimizer.py` runs a Bayesian optimization (BO) of the tracker layout described by
`GeoCLD.txt`. Every candidate geometry is written to a text file, its track
resolutions are computed with `RunMetrics.cc` (the `SolTrack::CovCalc` covariance
model above), and a scalar loss relative to the starting geometry is minimized.

### Setup

Everything (ROOT, a C++ compiler for ACLiC, numpy/uproot/matplotlib,
scikit-optimize, BoTorch/PyTorch) is provided by the pixi environment:

```
pixi install            # .pixi can be a symlink to a scratch area, e.g.
                        # ln -s /scratch/users/<user>/FastTrackCovariance/.pixi .pixi
pixi run optimize       # = python optimizer.py, with the options below
pixi run baseline       # test_baseline.py: evaluate the starting geometry once
pixi run short-run      # test_short_run.py: a very short optimization
```

ROOT 6.40 needs the `TF1` constructor fix in `SolGeom*.cxx` (already applied).
The magnetic field is still set in `SolGeomCLD.cxx` (`fB = 4`).

### Quick start

```
pixi run optimize --n-calls 700                    # fresh run, barrels + disks, muon collider
pixi run optimize --resume --n-calls 300           # continue it
pixi run optimize --collider HC                    # hadron-collider variant
pixi run optimize --no-disks --n-stations 4        # only the 4 VTX doublets
```

### Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--n-calls N` | 200 | new geometry evaluations in this run (baseline and resumed points excluded) |
| `--n-initial N` | 40 | quasi-random evaluations before the GP drives the search, counted over the whole (resumed) run |
| `--n-stations N` | all | optimize only the N innermost barrel stations |
| `--no-disks` | off | keep the disks fixed |
| `--collider {MC,HC}` | MC | MC: muon collider (nozzle, no beam pipe layer); HC: hadron collider (beam pipe, no nozzle) |
| `--resume` | off | continue from `opt/bo_checkpoint_<collider>.json`; without a checkpoint, start from `opt/GeoBEST_<collider>.txt` |
| `--seed GEO [GEO ...]` | – | add geometry files written by this optimizer as known starting points |
| `--backend {botorch,skopt}` | botorch | botorch: batches evaluated in parallel; skopt: one evaluation at a time |
| `--acq {ts,qlogei}` | ts | botorch batch selection: Thompson sampling in a trust region (fast) or q-LogEI (slow for large batches) |
| `--batch N` | = workers | geometries proposed per GP fit (botorch) |
| `--workers N` | 25% of the cores | parallel ROOT jobs and GP threads |
| `--min-hits N` | 6 | track-finding model: minimum measured hits per track |
| `--seed-hits N` | 3 | track-finding model: minimum hits in the seed subsystems |
| `--hit-eff E` | per subsystem | track-finding model: hit efficiency of every measurement layer |
| `--eff-weight W` | 1.0 | weight of the efficiency term in the loss (0 = resolution only) |

### What is optimized (parameters)

* **Barrel stations** (radius + length each). Barrel layers are grouped into rigid
  stations: a measurement layer, or a doublet of measurement layers closer than
  `DOUBLET_MAX_GAP`, together with the passive support layer(s) of the same label
  and length. A station moves as a whole (members keep their radial offsets).
  * radius: within `R_FRAC_LO`–`R_FRAC_HI` × baseline (±30%);
  * length: a fraction f ∈ [0, 1] of the longest allowed half-length,
    L = `L_MIN` + f·(L_max − `L_MIN`), where L_max clears the nozzle and every
    disk at the station's current radius. Overlaps are therefore impossible by
    construction.
* **Disk stations** (|z| each): only the +z disks are parameters, every −z disk
  mirrors its partner. Passive disks move with their sensor disk. |z| within ±30%
  of the baseline. When a disk moves outwards its inner radius grows to keep
  the baseline clearance to the nozzle (MC).

With the default geometry this is 10 barrel stations × 2 + 15 disk stations = 35 parameters.

**Never changed:** layer materials and thicknesses, resolutions, stereo angles,
number of layers, disk outer radii, the fixed passive layers (beam pipe, VTX
support shell at R = 0.113 m, ITK support tube at R = 0.570 m), the magnetic field.

### Constraints

All constraints are built into the parameter ranges, so every evaluated geometry is valid:

* **Ordering:** stations and disks cannot cross; neighbouring free stations keep at
  least `MIN_GAP` (2 mm) and split the space at the baseline midpoint. Disks only
  constrain each other when their radial extents overlap.
* **Fixed layers:** stations keep `MIN_GAP` from fixed passive layers; disks stay
  beyond the end of any fixed support shell they would cross.
* **Inner/outer tracker interface:** no station within `INTERFACE_GAP` (2 cm) of
  the fixed layers between the outermost ITK and innermost OTK station (the ITK
  support tube), so the interface stays free for services. The baseline violates
  this (ITK3 support at 1.1 cm): it stays the reference but is not a starting point.
* **Envelope:** nothing beyond the bounding box of the starting geometry
  (r ≤ 1.486 m, |z| ≤ 2.3 m, beam pipe excluded); the calorimeter sits outside.
* **Nozzle (MC only):** r_noz(z) = 0.1763·z (z < 1 m), 0.08474·z + 0.09156 m (z ≥ 1 m).

`barrel_nozzle_penalty` and `barrel_disk_overlap_penalty` are kept as safety
checks; they are zero for every geometry the optimizer generates.

### Collider modes (`--collider`)

* **MC** (muon collider): the tungsten nozzles replace the beam pipe as the
  innermost element. The `PIPE` layer is removed, barrel lengths and disk inner
  radii respect the nozzle, and the nozzle is drawn in the plots. It is a
  keep-out region, not material (tracks start at 10°, the nozzle angle).
* **HC** (hadron collider): beam pipe kept, no nozzle.

Each mode has its own baseline, checkpoint and best-geometry files.

### Track grid and loss

`RunMetrics.cc` evaluates, for every geometry, `N_THETA` = 40 polar angles
uniform in cos θ from 10° to 90° × `N_PT` = 20 transverse momenta log-spaced
from 0.5 to 100 GeV (800 tracks from the origin, φ = 0, z > 0 only; the
geometry is symmetric). The output tree `metrics` has, per track: `pt`,
`theta_deg`, `spt_rel` (σpT/pT), `sd0_um` (σd0 in µm), `sz0_um` (σz0 in µm),
`nmeas` and `mlay[nmeas]` (indices of the measurement layers crossed).

The resolutions are those of an ideal fit using every crossed layer, which is
what a Kalman filter + smoother would reach with perfect pattern recognition.
To stop the optimizer from removing layers for free, a **track-finding model** is added:

* a track is found if it has at least `SEED_HITS_MIN` = 3 hits in the seed
  subsystems (`SEED_LABELS` = VTX, VTXDSK, ITK, ITKDSK) **and** at least
  `N_HITS_MIN` = 6 measured hits in total;
* each crossed measurement layer gives a hit with probability `HIT_EFF[label]`
  (0.98 for all), so each track's finding efficiency follows exactly from the
  hit probabilities (Poisson-binomial) and varies smoothly with the geometry.

The loss, relative to the baseline on the same tracks (baseline = 1.0, lower is better):

```
r_t  = ( σpT/pT ratio + σd0 ratio ) / 2                      per track
res  = Σ eff_t · r_t / Σ eff_t                               efficiency-weighted resolution
loss = res + EFF_WEIGHT · (1 − <eff> / <eff>_baseline)
```

**These settings are placeholders.** With N_min = 6 and ε = 0.98, nearly every
track has enough redundant hits (baseline efficiency 99.7%), so the efficiency
term barely penalizes removing layers. It starts to matter from `--min-hits 8`
(baseline 97.3%, a layout with shortened ITK/OTK layers 94.5%). Tune N_min,
the seed subsystems and the hit efficiencies to the real reconstruction and
beam-induced background.

### Optimization backends

* **botorch** (default): each round fits one Gaussian process to all known
  points, on −log(loss) so that a few very bad geometries do not dominate, and
  proposes a batch of `--batch` geometries that are evaluated in parallel
  (each ROOT job in its own `opt/workers/wNN/` directory with its `root.log`).
  Batch selection (`--acq ts`) is Thompson sampling inside a TuRBO trust region
  around the best point, scaled per parameter by the GP length scales; the
  region doubles after `TR_SUCCESS_ROUNDS` improving rounds, halves after
  repeated failures and restarts when it collapses (`TR_*` constants). Proposals
  take about a second; a round is dominated by ROOT (~3.5 s).
* **skopt**: the original sequential GP + expected improvement. The GP refit
  gets slow beyond a couple of hundred points.

**Resources:** at most `CPU_FRACTION` = 25% of the node is used, both for the
parallel ROOT jobs and for the GP's threads (`--workers` overrides).

### Checkpoints, resume and seeds

* Every evaluation is saved to `opt/bo_checkpoint_<collider>.json` (written
  atomically after each round), together with the configuration it belongs to
  (collider, free parameters, track grid, loss settings).
* `--resume` loads it and gives all points to the optimizer without re-running
  ROOT; `--n-calls` counts the additional evaluations. It refuses checkpoints
  written with a different configuration and drops points outside the current
  bounds. Without a checkpoint it starts from `opt/GeoBEST_<collider>.txt`
  (or `opt/GeoBEST.txt`).
* A fresh run moves an existing checkpoint to `bo_checkpoint_<collider>.prev.json`.
* `--seed GEO` adds any geometry file written by this optimizer as a known point.

### Outputs (`opt/`)

| File | Content |
|---|---|
| `GeoBEST_<collider>.txt` | best geometry found (GeoCLD format) |
| `geometry.png` | plot of the best geometry |
| `geometry_start.png` | plot of the starting geometry |
| `bo_checkpoint_<collider>.json` | all evaluations (parameters and losses) |
| `GeoBASE_<collider>.txt`, `metrics_baseline_<collider>.root` | baseline geometry and its metrics (reference of the loss) |
| `GeoOPT.txt`, `metrics.root` | last sequential evaluation |
| `workers/wNN/` | per-worker geometry, metrics and ROOT log (parallel runs) |

Plots: active barrels blue, active disks red (solid); passive layers grey dashed
with width ∝ X/X0; nozzle grey (MC); envelope green dotted; interface keep-out
band green; a side panel zooms on the vertex region. Any geometry can be drawn with
`pixi run python -c "import optimizer as o; o.draw_from_file('opt/GeoBEST_MC.txt')"`.

### Tunable constants (top of `optimizer.py`)

| Constant | Default | Meaning |
|---|---|---|
| `N_THETA`, `THETA_MIN_DEG`, `N_PT`, `PT_MIN`, `PT_MAX` | 40, 10°, 20, 0.5, 100 GeV | track grid |
| `DOUBLET_MAX_GAP` | 5 mm | sensors closer than this form one station |
| `MIN_GAP` | 2 mm | minimum clearance between stations / disks / fixed layers |
| `INTERFACE_LABELS`, `INTERFACE_GAP` | (ITK, OTK), 2 cm | inner/outer tracker keep-out band |
| `R_FRAC_LO`, `R_FRAC_HI` | 0.7, 1.3 | barrel radius range relative to baseline |
| `Z_FRAC_LO`, `Z_FRAC_HI` | 0.7, 1.3 | disk \|z\| range relative to baseline |
| `L_MIN` | 5 cm | minimum barrel half-length |
| `CLEARANCE` | 1 mm | clearance to the nozzle and to disks |
| `N_HITS_MIN`, `SEED_HITS_MIN`, `SEED_LABELS`, `HIT_EFF`, `EFF_WEIGHT` | see above | track-finding model and loss |
| `FAIL_LOSS` | 3.0 | loss of a failed evaluation |
| `CPU_FRACTION` | 0.25 | share of the node's cores used |
| `TS_CANDIDATES`, `TR_*` | | Thompson-sampling / trust-region settings |

The envelope (`R_ABS_MAX`, `L_ABS_MAX`, `Z_ABS_MAX`) is recomputed from the
starting geometry at start-up.

### Known limitations

* Resolutions assume an ideal fit; pattern recognition is only modelled through
  the placeholder hit requirements above. There is no beam-induced background,
  occupancy, fake-track or timing model yet.
* No cost term (silicon area, material budget, channels): lengths and radii tend
  to go to the edges of their ranges where the loss allows it.
* Layer counts, materials and sensor resolutions are not optimized.
