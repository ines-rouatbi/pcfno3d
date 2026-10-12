# Tukhnanutha — Project Objective & Alignment Check

This file exists so any Claude Code session opened in this repo starts from the actual business
objective, not just the code. `CLAUDE.md` documents *what the code does*; this file documents
*what it's for* and *whether current work still points at it*. Update the "Current state" section
whenever the direction changes — don't let it go stale.

## 1. Context

PFE (ESPRIT final-year thesis) by Ines ROUATBI, hosted by **Tukhnanutha** (Bizerte; rep. Rayen Ben
Amor), 05-03-2026 → 05-09-2026. Part of Tukhnanutha's **ANEO** ecosystem.

## 2. Official objective (from `ANEO_Presentation.pptx`)

**General objective**: build a **Physics-Constrained FNO** neural operator that reconstructs the
3D wind field (u, v, w) over Tunisia from the sparse ANEO sensor network (measurements at
z≈25–30m only), extrapolated up through the atmospheric column to **1000m altitude**, producing
~2.5M virtual 3D nodes. Intended end use: near-real-time input to a **3D visualization / web
application**.

- OS1 — Spatial reconstruction: learn the horizontal field from sparse ANEO measurements.
- OS2 — Physical constraints: enforce anelastic mass conservation ∇·(ρv)=0 in the loss.
- DSO1 — RMSE(wind) ≤ 0.5 m/s at measured points.
- DSO2 — mass-conservation residual ≤ 1e-4.
- DSO3 — spatial smoothness score.
- DSO4 — inference < 1s/timestamp.

**Scope directive (Rayen, 2026-07-17): reconstruct the full atmospheric state, not wind-only.**
Every model currently outputs only `wind_u_m_s`, `wind_v_m_s`, `wind_w_m_s` even though it's fed
7 physical variables as input (`temperature_c`, `air_density_kg_m3`, `relative_humidity_pct`,
`pressure_pa`, plus u/v/w — see `NUMERIC_INPUT_COLS` in `fno2D.ipynb` cell 3). The rule going
forward: **whatever variable is an input must also be an output** — the reconstructed field
should cover temperature, density, humidity, and pressure alongside wind, not just wind. This is
a scope correction, not a new feature request; treat any model whose target set is narrower than
its input set as incomplete against the objective.

## 3. Data reality check (important constraint, easy to miss)

> **Superseded on 2026-09-28 — read §5 first.** The `aneo_dataset_3d_v2` generator run emits
> **virtual 3D nodes at all 12 levels** (849,184 nodes, 25–1000 m ASL), so the constraint below no
> longer binds: a 3D-supervised operator *is* trainable against that dataset. The paragraph is kept
> because it still describes the *physical* sensor network (one real altitude per point) — the 3D
> targets are generator output, not measurements, which is what the mass-conservation and
> ground-condition penalties exist to keep honest.

The ANEO grid (`aneo_grid_points.csv`) stores **one altitude per point** (`z_rel_m` ≈ 25–30m
AGL). There is no ground truth at 100m/500m/1000m. This means:

- A "native" fully 3D-supervised operator (learning u/v/w as a function of x, y, *and* z from
  data) **cannot be trained** — there's nothing to supervise it against above sensor height.
- The 0–1000m column can only be produced by combining (a) a model that reconstructs the
  horizontal field well at sensor height, with (b) a **physics-based vertical extrapolation**
  (log-wind/power-law profile + mass conservation) projecting that field upward.
- This is *not* a shortcut — it's the correct design given the data. "FNO 3D" in the deliverable
  name means "physics-constrained operator whose output is used to build a 3D field," not
  "operator with a literal 3D spectral convolution."

**Separate, related gap flagged by Rayen (2026-07-17): the horizontal grid itself isn't flat.**
Grid points are ~500m–1km apart in the real ANEO network (the `data_4x` synthetic dataset uses a
denser 50m raster over a 50km×25km test region, but `z_terrain_m` there already spans 0–526m,
σ≈109m — so even at that spacing, adjacent cells sit at meaningfully different altitudes because
Tunisian terrain isn't flat). Confirmed in code (`fno2D.ipynb`, `PhysicsConstrainedLoss.forward`):
the mass-conservation term computes only `div = d(ρu)/dx + d(ρv)/dy` with a constant `dx=50.0` —
a flat-plane 2D divergence. Two concrete problems:
1. `∂(ρw)/∂z` is entirely absent from `div`, even though the model predicts `w_pred` and applies a
   `w=0` ground boundary condition two lines later. The anelastic constraint ∇·(ρv)=0 (DSO2) is
   being scored against an incomplete equation.
2. No terrain-slope/metric correction — `z_terrain_m` is only used as a normalized static *input*
   feature (`alt_grid`), never inside the physics loss, so a step of `dx` in x/y is silently
   treated as a step at constant altitude when it's actually a step along sloped terrain.
This is the same forked `PhysicsConstrainedLoss` pattern the other 7 notebooks copy, so likely
affects all of them, not just `fno2D.ipynb`. Not yet fixed — recorded here so DSO2 numbers from
any of these notebooks are read as provisional until the divergence includes the vertical term and
a terrain correction.

## 4. Planned timeline (Gantt, Mars–Septembre 2026)

| Task | Content | Planned month |
|---|---|---|
| T1 | Understand wind dynamics | Mars |
| T2 | Data prep & analysis | Avril |
| T3 | Design FNO 3D | Mai |
| T4 | Physics constraints | Juin |
| T5 | Training | Juin |
| T6 | 3D reconstruction + visualization | Juillet |
| T7 | Performance evaluation | Juillet |
| T8 | Visualization & interpretation | Juillet–Août |
| T9 | Deployment & tests (→ web app integration) | Août |

## 5. Current state (snapshot as of 2026-10-11)

**The drift flagged in July is resolved.** The project converged on a single architecture, the
dataset was regenerated with true 3D targets, and the work has moved into T6–T9.

What changed since July:

- **One architecture, actually 3D.** `notebooks/current/fno3d_v7_1.ipynb` (dated 2026-10-01, v2→v7
  iterated between Sep 25 and Oct 1) is a genuine **PC-FNO3D**: `SpectralConv3d` over (z, y, x),
  25 input channels, output u/v/w on 12 ASL levels × 250 × 500 (~1.5 M nodes per instant),
  ~453 M parameters. The 2D fork and the 6 competing architectures are closed — moved to
  `notebooks/archive/`.
- **Dataset regenerated** (`aneo_dataset_3d_v2`, 2026-09-28): 100 days, 1000 instants, surface grid
  1000×500 at 50 m, 3D nodes 500×250 at 100 m on levels
  25/50/75/100/150/200/300/400/500/650/800/1000 m ASL. Day-level train/val/test split (70/15/15) in
  `split_days.json`. Generator QC (`qc/qc_summary.json`) passes every check — mass residual max
  6.4e-12, spatial/temporal correlations, hydrostatic error p99 — **except** direction-sector
  balance (0.095 vs 0.10 threshold), i.e. the 0° sector is slightly under-represented. Worth a line
  in the report; not a blocker.
- **The vertical dimension is now supervised, not extrapolated.** The `reconstruct_3d` post-hoc
  step that §3 was built around is obsolete: the operator predicts the column directly and BLOC 12
  reconstructs a full instant in one forward pass.
- **DSO targets are encoded in code, not just slides**: DSO1 (RMSE ≤ 0.5 m/s and R² ≥ 0.95 per
  level, u/v — plus an extended R² ≥ 0.95 on w), DSO2 (mass residual RMS ≤ 1e-4), DSO3 (roughness
  ratio ≤ 1.1), DSO4 (< 1 s per instant), DSO5 (RMSE ≤ 0.7 m/s with 25 % of sensors). Evaluation
  runs on **synoptic instants**; fast episodes are reported separately as a stress test.
- **Deliverables drafted** (2026-10-01/02): `docs/presentations/PC-FNO3D evaluation.pptx`,
  `PC-FNO3D recontruction part2.pptx`, `PC-FNO3D deployment part3.pptx`, a French version, plus
  `docs/PC-FNO3D notions du projet et rôle de chaque bloc.pdf` and `Tenseurs du notebook
  PC-FNO3D.pdf`.
- **Infrastructure built 2026-10-11** (this repo's `src/pcfno3d/`): the notebook is ported
  block-by-block into an installable package with a CLI, YAML configs for local/Colab/test, a
  FastAPI inference service, a Dockerfile, a web export format + viewer, a synthetic mini-dataset
  generator, and 30 CPU tests. The whole pipeline — cache → train → DSO evaluation → 3D
  reconstruction → web export — was run end to end on the mini-dataset. See `README.md`.

### v7 measured results (from `notebooks/current/fno3d_v7.ipynb`, Colab, 2026-10-01)

The first real DSO numbers on record. Model `best_dso_v7.pt` = the BLOC 9 model (epoch 140; the 9b
fine-tune never beat it). 120 synoptic test instants, 50 % of sensors visible.

| DSO | target | v7 | verdict |
|---|---|---|---|
| DSO1 RMSE u, v per level | ≤ 0.5 m/s | max **u 0.665** (1000 m), v 0.508 (1000 m) | **fails** |
| DSO1 global RMSE | ≤ 0.5 m/s | u **0.542**, v 0.437 | **fails** (u) |
| DSO1 R² u, v (and w) | ≥ 0.95 | min 0.991 (w min 0.966) | passes |
| DSO2 mass residual | ≤ 1e-4 | 2.58e-5 | passes |
| DSO3 roughness ratio | ≤ 1.1 | 0.951 | passes |
| DSO4 inference | < 1 s | 91 ms | passes |
| DSO5 RMSE at 25 % sensors | ≤ 0.7 m/s | 0.646 | passes |

So **v7 meets five objectives out of six**. The single failure is DSO1, and it is localised:
every level up to 500 m passes (u ≤ 0.481, v ≤ 0.456); u then rises 0.564 (650 m) → 0.632
(800 m) → 0.665 (1000 m), and v crosses at 1000 m only. The WMO-style criterion
(RMSE ≤ max(0.5, 10 % of mean speed)) **passes** — in relative terms the upper-level error is
small; the fixed 0.5 m/s bites exactly where the wind is strongest. That is worth raising with
Tukhnanutha as a property of the target, not only of the model.

Diagnosis, from the training log and the evaluation tables:

- **Not overfitting.** Train and validation data loss end level (0.0050 vs 0.0053), flat from
  epoch ~129 as the OneCycle schedule reaches zero. More regularisation would not help.
- **Objective mismatch.** Training minimises a *mean* error over all nodes; DSO1 is a *maximum*
  over levels. A model can have an excellent mean and still fail its worst level — which is what
  happened. The v7 already corrected this for w (`w_level_w`) but not for u and v.
- **Sensor-count anomaly.** v7 is *worse* with all sensors than with half (max u RMSE 0.705 at
  100 % vs 0.593 at 50 %). Training draws coverage uniformly in [0.2, 1.0], so 100 % is an edge
  case — yet it is the production case, and the API defaulted to it. BLOC 12's single instant at
  100 % shows u RMSE 1.335 at 1000 m.
- **The 9b result was confounded.** It is recorded as "fine-tuning degrades", but 9b also doubled
  `lambda_grad` (0.15 → 0.3), which explicitly trades RMSE for roughness. It does not show that
  fine-tuning cannot help.
- **The BLOC 10c ensembling was never measured** — interrupted (cell `model = _model_single`), and
  it reused the full-sensor geostrophic features on masked inputs, which put every extra pass out
  of the training distribution.

Two reporting issues found and fixed in the package (verdicts unchanged): BLOC 12 printed R² =
−11 899 at 25 m — a 0 %-coverage level admitted by a "≥ 10 nodes" test; the notebook labels DSO3
as "0.9–1.1" while only checking ≤ 1.1 (0.951 passes either reading).

### Corrections applied (2026-10-11) — the v8 recipe

Implemented in the package, all opt-in so v7 stays reproducible; `configs/colab_v8.yaml` turns
them on; `notebooks/current/pcfno3d_v8_colab.ipynb` runs them **cheapest first**:

0. **Validate the port** by re-evaluating `best_dso_v7.pt` with the package and comparing to the
   notebook's own `eval_best_dso_v7/test_par_niveau_synoptique.csv`. If it does not reproduce v7
   within 0.01 m/s, stop — every v7/v8 comparison would be meaningless.
1. **Ensembling without retraining** (`evaluate --tta 8`): 8 passes on different sensor subsets,
   features *recomputed per pass*. Directly targets the sensor-count anomaly. Must stay under 1 s
   (8 × 91 ms ≈ 0.73 s before overheads) — the evaluation measures DSO4 in that mode.
2. **Level weights for u, v** (`loss.uv_level_weight`), w = (RMSE_k / 0.5)², clipped, normalised
   so L_data keeps its scale; **measured on validation only** — the config refuses a weights file
   marked `test`.
3. **Full-coverage examples** (`mask.p_full: 0.15`).
4. **Fine-tune from v7, selected on the DSO score** (`train.init_from`, `train.select_by: dso`),
   with `lambda_grad` left at 0.15. The original model is the first candidate, so fine-tuning can
   only keep v7 or improve on it.

All of it is tested on the mini-dataset (72 tests). **None of it is verified to reach DSO1** —
that takes the v7 weights and a GPU, i.e. running the v8 notebook on Colab (~1 h 20). If neither
v7 + ensembling nor v8 passes, the upper levels are likely information-limited: sensors sit at
25–30 m, and the free atmosphere is tied to them only through a geostrophic wind that is a
*single plane fit* over the whole domain. The next lever would be a spatially varying
geostrophic estimate — an input change requiring full retraining.

### v7 served locally (2026-10-12)

- **The right v7 is `notebooks/current/fno3d_v7.ipynb`** (confirmed by the user). Checked block by
  block, comments stripped: its model code is **identical** to `fno3d_v7_1.ipynb`, the commented
  copy the package was ported from. The only difference is `MAP_ALTITUDE` (1000 m vs 120 m), a
  plotting parameter; the package now defaults to 1000 m as well — on the real domain the ground
  reaches 677 m, so 120 m ASL is underground over much of the map.
- **The real weights are installed** (`AppData/Local/pcfno3d/runs/v7`, sha256 `b2529442…`) and
  served on port 8000 with `-c local_v7`. The installer confirmed epoch 140 and selection score
  1.19523 — the model the notebook evaluated — and **all 453.3 M parameters loaded under
  `strict=True`**: the port is structurally identical. `norm_stats.json` matches BLOC 5's printout
  exactly.
- **One real test instant reconstructed** (`017_120000`, all sensors): u 0.09–0.24, v 0.11–0.31,
  w ≈ 0.01 m/s RMSE across levels. A sanity check, not a validation: that instant is calm
  (1–2 m/s at 300 m), and absolute error grows with wind speed.
- **CPU only**: 15–25 s per instant here vs 91 ms on Colab's GPU, so DSO4 is not met locally.
  Weights are memory-mapped (+0.03 GB RAM at load instead of ~3.6 GB), and the service computes
  each instant once even when the web app sends parallel requests.

### v8 notebook, first run (Colab A100, 2026-10-12)

- **Step 0 passed exactly**: the package re-evaluated `best_dso_v7.pt` on the 120 synoptic test
  instants and matched the notebook's BLOC 10 table at **all 12 levels with a 0.0000 m/s maximum
  deviation** (u 0.665 / v 0.508 at 1000 m, global u 0.542). Every number the package produces
  is now comparable to the v7 notebook.
- **Step A (8-pass ensembling, keep 0.7) is rejected**: worse at every level — 1000 m u 0.672,
  global u 0.579 (vs 0.542), DSO5 0.684 (vs 0.646), and DSO4 930 ms, close to the 1 s limit.
  Likely cause: under the DSO1 protocol (50 % of sensors visible) each pass keeps 70 % of those,
  i.e. sees only 35 % of the sensors; averaging does not give back the information dropped.
  Only at 100 % sensors does it help (robustness sweep: u max 0.705 → 0.615, v 0.516 → 0.482) —
  relevant to serving with all sensors, not to the DSO1 verdict, and 8x the compute.
- **Step B (validation)**: score 1.193; out of threshold only at 800 m (u 0.572, v 0.562) and
  1000 m (u 0.557, v 0.597) — 650 m passes on validation (0.484). Weights: 1.32 at 800 m, 1.43 at
  1000 m, 0.72–0.95 at 100–650 m.
- The cache was rebuilt from the dataset in 19 min on that runtime (estimated 1 h).
- **Step C (fine-tuning, 25 epochs, 147 s each)**: validation DSO score 1.193 -> **1.183**
  (epoch 25), i.e. < 1 % better. Validation data loss flat at 0.0053 from epoch 11 while train
  loss kept falling (0.0052 -> 0.0046): the model learns nothing more that generalises.
- **Step D, v8 single pass on test (120 synoptic instants)**: DSO1 still **fails**. Upper
  levels: 650 m u 0.562 / v 0.424, 800 m u 0.623 / v 0.489, 1000 m u 0.649 / v 0.529 (v7: 0.564 /
  0.408, 0.632 / 0.466, 0.665 / 0.508). u improves by <= 0.016, v gets worse by up to 0.023;
  global u 0.535 (v7 0.542), v 0.452 (v7 0.437). Vector RMSE at 1000 m is identical (0.837 both).
  DSO2 2.50e-5, DSO3 0.950, DSO4 104 ms, DSO5 0.641: still met. **Net: v8 is not better than v7.**
- **Conclusion of the v8 campaign**: neither ensembling nor fine-tuning with level weights,
  full-coverage examples and DSO selection moves the 650–1000 m error meaningfully. That supports
  the information-limit hypothesis below (one domain-wide geostrophic plane is the only link
  between 25–30 m sensors and the free atmosphere). v7 stays the served model.

### Still not verified

- **Whether a spatially varying geostrophic input can reach DSO1** — untested; needs a full
  retraining (~140 epochs x ~145 s ≈ 5–6 h of A100).
- T9 is **scaffolded, not deployed**: API, MCP, Docker and the Colab tunnel exist and are tested,
  and v7 runs locally, but nothing is hosted, and there is no CI or monitoring.

Next concrete steps, in order: (1) decide with Tukhnanutha between reporting v7 against the OMM
criterion (met) and funding an input-architecture change for DSO1; (2) serve that configuration via the Colab notebook
and `pcfno3d link set`; (3) write the report around the archived `test_summary.json`.

## 5bis. Previous snapshot (2026-07-17) — kept as history

- T1/T2 landed 2–3 months late (EDA/cleaning notebooks dated July 9; ANEO data generated June 1
  instead of April).
- T3–T5 forked into **6 competing 2D architectures** (FNO2D ×3 variants, UNet-FNO, GNN, DeepONet,
  ResNet/ViT baselines) instead of converging on one model — see `CLAUDE.md`.
- The **vertical extrapolation step exists but was left behind**: an older pipeline
  (`aneo_physical_db_multitime_v3_progress/.../data_4x/ANEO_FNO2D_Model.ipynb` +
  `05_evaluation.ipynb`, dated June) has a working `reconstruct_3d` function producing 7-level
  (25–1000m) U/V/W fields with logged MAE/R² (R² for u ≈ 0.80–0.98 — but see the re-run below,
  which shows this is checkpoint-dependent and should not be assumed current). None of the 8
  newer top-level notebooks call or reproduce this step — confirmed by grepping all of them for
  `reconstruct_3d` / `z_levels` (zero matches).
- **`05_evaluation.ipynb` was re-run 2026-07-17**, but against `models/best_model.pth` — the old
  June FNO2D checkpoint at **epoch 2 only** (val loss 0.072) — not any of the 6 newer
  architectures. Fresh numbers (`figures_evaluation/metrics_summary.csv`, 28 test snapshots):
  global RMSE u=0.453, v=0.675, w=0.014, R² u=0.002, v=0.804, w=0.914; R² u drops to **-0.578** in
  the calme regime. Against DSO1 (RMSE ≤ 0.5 m/s), u passes but v fails in every regime. This is
  materially worse than the "0.80–0.98" figure above — that number came from a different, more
  converged run, and the epoch-2 checkpoint should not be treated as representative of the
  pipeline's ceiling.
- A Drive export dumped the same day
  (`aneo_physical_db_multitime_v3_progress/.../data_4x/v1/*.zip`) shows the 6-architecture
  benchmark is further behind than it looks: only DeepONet (`best_deeponet_model.pth`, saved
  Jul 14) and ResNet (`best_convresnet_model.pth`, saved Jul 15) have a saved checkpoint. The new
  FNO2D, ViT, and GNN folders in that export are empty — no checkpoint, no figures — meaning
  those three either failed during that training job or never got that far. "6 competing
  architectures" currently means 2 trained candidates, not 6.
- T6, T7, T8, T9 have not started for the current generation of models: no head-to-head DSO1–DSO4
  evaluation across the 6 architectures (the one evaluation run that exists targets the old June
  checkpoint, not the new ones), no visualization code, no API/backend, no deployment or
  web-integration code anywhere in the repo.

**Scan of `data_4x/Colab Notebooks-20260717T220940Z-1-001/` (2026-07-17, ~22:10) — supersedes the
"6 architectures = 2 trained candidates" read above.** This is a Google-Drive-synced copy of 4
notebooks (`fno2D.ipynb`, `vit.ipynb`, `DeepOnet.ipynb`, `ResNet.ipynb` — no GNN) with their
**executed outputs still embedded**, exported ~20min after the `v1/` zip snapshot used above.
Reading the actual execution logs (not just checkpoint files on disk) changes the ranking:

| Model | Test-set R² (u / v / w) | RMSE u / v (m/s) | Notes |
|---|---|---|---|
| **FNO2D (new)** | **0.987 / 0.989 / 0.997** | not directly printed, but MAE_u≈0.03–0.16, MAE_v≈0.11–0.22 | Resumed from an epoch-2 checkpoint, trained 150 more epochs, best at epoch 124, R²avg=0.9912. By far the strongest result seen anywhere in this project. |
| ResNet | 0.969 / 0.981 / 0.993 | 0.146 / 0.259 | Passes DSO1 (≤0.5 m/s) clearly. |
| DeepOnet | 0.913 / 0.964 / 0.914 | 0.239 / 0.356 | Passes DSO1, weaker than ResNet. |
| ViT (new) | 0.618 / 0.884 / 0.962 | not printed | R²avg=0.8225 — confirms baseline-only status. |
| FNO2D (old, epoch 2) | 0.002 / 0.804 / 0.914 | u=0.453, v=0.675 | Superseded — this is the checkpoint `05_evaluation.ipynb` was scored against; not representative. |

**FNO2D (new) is the clear frontrunner** — this effectively answers step 2 below, *if the weights
can be recovered*. That's the catch:

**Checkpoint declared lost (Rayen, 2026-07-17), then recovered (2026-07-18).** `fno2D.ipynb`'s
`MODEL_DIR` is `Path("/content/aneo_work") / "aneo_models"` — Colab's local ephemeral disk, wiped
on disconnect — with no `shutil`/Drive-copy step anywhere in the notebook at the time (confirmed
by grep). Fixed in the repo copy of `fno2D.ipynb` on 2026-07-17: added `DRIVE_MODEL_DIR = DATA_DIR
/ "models"` and a `shutil.copy2` after every best-checkpoint save plus one at end of training, so
this can't happen again going forward.

**Update 2026-07-18: the checkpoint itself was recovered**, not re-trained. Rayen added
`data_4x/aneo_backup/aneo_backup/` (2GB: `aneo_models/best_fno2d_model.pth` 845MB,
`aneo_tensors/` 1.2GB of cached `.npy` day/hour tensors, `aneo_output/` metrics CSVs,
`aneo_figures/`). Loaded and verified directly (`torch.load`, weights_only=False):

- `epoch=129`, `val_loss=0.01545`, `r2=0.9903` (`r2_u=0.9849`, `r2_v=0.9890`, `r2_w=0.9972`) — this
  is the same run referenced in the "checkpoint declared lost" entry above (R²avg 0.9903 vs. the
  0.9912 read off the execution log — same league, likely a couple epochs later at early-stop).
  Checkpoint dict keys (`r2_u`, `val_l_mass`, `val_l_bc`, …) match exactly what the current
  `fno2D.ipynb` cell 13 writes, so it loads with cell 14 as-is.
- `aneo_output/fno2d_evaluation_results.csv` (24 test snapshots, produced by cell 14's per-snapshot
  loop) — **DSO1 passes cleanly across the board**: RMSE u ∈ [0.034, 0.211], RMSE v ∈ [0.113, 0.269],
  both always ≤ 0.5 m/s. This is the first evidence in the project of DSO1 passing on *any*
  checkpoint of the new architecture (the epoch-2 checkpoint `05_evaluation.ipynb` scored earlier
  failed v in every regime — see below, unchanged).
- `aneo_output/reconstruction_3d_results.csv` also exists (24 rows, `ws_25m/100m/300m/1000m` per
  day/hour) — **but no cell in the current local `fno2D.ipynb` produces this file** (grepped for
  `reconstruct_3d`/`Z_LEVELS`/`reconstruction_3d`: zero matches; cell 14 only writes
  `fno2d_evaluation_results.csv`, cell 16 is the single-snapshot PyVista viz). This must have come
  from an extra cell run in Colab that never made it back into the local notebook copy, or a
  different script entirely — **provenance unconfirmed, ask Rayen before trusting these numbers or
  treating step 3 below as done**. The old 7-level `reconstruct_3d` in `ANEO_FNO2D_Model.ipynb` is
  unrelated (still un-ported, still targets the stale epoch-2 checkpoint per `out.txt`/`out2.txt`).

Decision going forward: don't re-run FNO2D training (old step 2) — this checkpoint is usable now.
Step 6 renumbered below accordingly.

**Full-repo rescan (2026-07-17, evening)** confirmed the above and added:
- `fno2D.ipynb` was edited today at 22:20, right after `05_evaluation.ipynb` (22:18) and just before
  the `v1/` export zips (22:44–22:49) — the DeepOnet/ResNet-only checkpoint situation described
  above is corroborated, not new. Active work today is concentrated in `data_4x/` (model
  export/evaluation) and repo root (`fno2D.ipynb`).
- **Data hygiene, not urgent but worth a look**: `data_4x.rar` reports as ~10.5GB, wildly out of
  line for this dataset's real size — likely corrupt or a placeholder; don't trust it as a backup
  without checking. Full-dataset lineage is scattered across `data_full/` (complete, Apr 27),
  `data_claude3.rar`/`data_claude4.rar` (unextracted, presumably days ~1-21, unverified), and
  `data_claude5/`/`data_claude6/` (4 days each, Jun 1, partial) — four partial/uncertain sources
  instead of one canonical one.
- No `.pth`/`.pt`/`.ckpt`/`.h5` files exist anywhere outside `data_4x/models/best_model.pth` and
  whatever's bundled inside the unopened `v1/` zips — confirms "6 architectures" is mostly notebook
  code, not trained models.

**Read**: behind schedule, and drifting from "one physics-constrained operator → extrapolate →
visualize → deploy" toward "benchmark many 2D architectures" with no path back to the 3D output
currently wired up.

## 6. Realignment plan — how to get back on the stated objective

> **Mostly executed as of 2026-10-11** (see §5): the convergence on one architecture and the move
> to a genuinely 3D supervised operator are done. What remains from this plan is the deployment and report
> half — kept below for the steps still open.

With ~7 weeks left (internship ends 05-09-2026), the fastest path back to the objective is to
**stop adding architectures and reconnect the pipeline that already worked**, rather than build
a bigger model. Ordered by dependency — each step assumes the ones above it are done:

0. **Done (2026-07-17/18).** Drive checkpoint save added to `fno2D.ipynb` (`shutil.copy2` to
   `DRIVE_MODEL_DIR` on every best-epoch save + end of training), and the checkpoint itself was
   separately recovered by Rayen and verified (`data_4x/aneo_backup/aneo_backup/aneo_models/
   best_fno2d_model.pth`, epoch 129, R²avg=0.9903, DSO1 passes on all 24 test snapshots). **No
   re-training needed** — architecture selection is effectively decided (R²avg=0.99 vs ResNet 0.98
   and DeepOnet 0.93) and a usable, DSO1-passing checkpoint now exists. Before using it further,
   copy `best_fno2d_model.pth` into wherever the working `fno2D.ipynb` session's `MODEL_DIR` points
   (its checkpoint dict already matches what cell 13/14 expect). GNN still has no evidence of ever
   completing a run and stays out of scope; don't train a 7th architecture.
1. **Fix the mass-conservation term before trusting any DSO2 number.** Current `L_mass` in
   `PhysicsConstrainedLoss` (`fno2D.ipynb` and its forked siblings) is a flat 2D divergence —
   missing `∂(ρw)/∂z` entirely and no terrain-slope correction for `dx` (see Section 3). Any
   DSO2 result computed before this fix is measuring the wrong equation, including the
   `val_l_mass` numbers already logged in the recovered checkpoint's history. Do this first because
   step 4 (DSO evaluation) is worthless until it's fixed, and it's a localized change (one loss
   class) that doesn't block step 3.
2. **Re-attach vertical extrapolation to the winner.** Port `reconstruct_3d` (7 levels, 25–1000m)
   from `ANEO_FNO2D_Model.ipynb` onto the recovered FNO2D checkpoint. This is what actually
   produces the "3D" deliverable — it's largely done, just disconnected from current work. (Do
   not confuse this with the unverified `reconstruction_3d_results.csv` found in the recovered
   backup — that's a 4-level wind-speed-only file of unknown provenance, not this step.)
3. **Widen the target set to match the input set** (scope directive, Section 2) on the winning
   architecture, before running the DSO evaluation below — otherwise the numbers only ever
   describe wind, not the full atmospheric state the objective now requires. In `fno2D.ipynb`:
   set `TARGET_COLS` to all of `NUMERIC_INPUT_COLS` (u,v,w first so physics-loss channel indices
   0/1/2 stay valid), add 4 output heads (temp/density/humidity/pressure) alongside `head_u/v/w`,
   regenerate the cached `Y.npy` tensors (currently 3-channel — add a target-set check keyed on
   `split.json` so stale 3-channel caches aren't silently reused), and make sure the wind-gust
   augmentation in `ANEODataset` (`Y *= scale`) only scales wind channels, not the new ones.
4. **Run DSO1–DSO4 evaluation on the combined model** (fixed loss + FNO2D + vertical extrapolation
   + full target set), not on the bare 2D wind-only reconstruction. RMSE and mass-conservation
   residual should be measured on the extrapolated 3D field, since that's what the objective and
   the web app will consume.
5. **Start T6/T8 (visualization) and T9 (deployment) in parallel with step 4**, not after — zero
   code exists for either. Minimum viable slice: a script that exports one reconstructed
   timestamp (U3D/V3D/W3D + the 4 new fields) to a format the web app can consume (GeoJSON/binary
   grid + a small serving endpoint), so the integration contract gets validated early.
6. **Freeze architecture scope.** Treat ResNet/ViT as baselines-only, not final-deliverable
   candidates, so effort stops splitting further.
7. **Data hygiene, low priority, do opportunistically.** Sanity-check `data_4x.rar` (~10.5GB,
   likely corrupt/placeholder) before relying on it as a backup; consolidate the scattered
   `data_claude3/4/5/6` lineage into one canonical source instead of leaving 4 partial copies.

## 7. Keeping this file useful

Whoever picks step 1–5 above should update Section 5 with the outcome (which architecture won,
whether the vertical extrapolation was re-attached, what the new DSO numbers are) so the next
session — or the next person opening this repo — isn't re-deriving this from file timestamps
again.
