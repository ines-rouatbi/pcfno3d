# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Before doing any planning or direction-setting work here, read `tukhnanutha.md`.** It has the
actual business objective (Tukhnanutha/ANEO), the DSO targets, the planned Gantt timeline, and a
live "current state vs. plan" section — this file below only documents what the code does, not
what it's supposed to accomplish or whether current work still points at it.

## What this is

A PFE (final-year engineering thesis) project on physics-informed deep learning for **wind field
reconstruction across Tunisia**. The work converged (September–October 2026) on a single
architecture, **PC-FNO3D**: a 3D Fourier Neural Operator that takes one instant of sparse surface
measurements and outputs u, v, w over 12 altitude levels (25–1000 m ASL).

Two things live side by side:

1. **`src/pcfno3d/`** — the pipeline as an installable package (`pip install -e .`), ported
   block-by-block from the reference notebook. This is what trains, evaluates, serves and gets
   tested. Entry point: `python -m pcfno3d <command>`.
2. **`notebooks/current/fno3d_v7_1.ipynb`** — the reference Colab notebook (BLOC 0 → BLOC 12),
   kept for experimentation. **The package is the port of this notebook; when you change the
   physics or the architecture in one, mirror it in the other** (the mapping table is in
   `README.md`). `fno3d_v7.ipynb` is the same pipeline **with its Colab outputs** —
   the only record of real v7 DSO numbers. Read its BLOC 9/10/12 outputs before claiming
   anything about model quality; `tukhnanutha.md` §5 summarises them.

**`fno3d_v7.ipynb` is THE v7** (user-confirmed). Its model code is identical block by block to
`fno3d_v7_1.ipynb` — the commented copy the package was ported from — so the port tracks the
right version; only `MAP_ALTITUDE` (a plot setting) differs. Re-run that comparison
(comments stripped) before assuming any other notebook matches.

**Serving v7 locally:** `configs/local_v7.yaml`, weights installed by
`scripts/install_v7_weights.py` into `AppData/Local/pcfno3d/runs/v7` (outside OneDrive — 1.81 GB).
On CPU, `WindFieldModel` memory-maps the checkpoint and builds the model on the `meta` device
with `load_state_dict(assign=True)`: loading costs ~0 RAM. **torch 2.2 requires a `str` path for
`mmap=True`** — a `Path` raises. CPU inference is 15–25 s per instant, hence the compute lock in
`WindFieldModel.predict` and the double-checked caches in `service/app.py` and `mcp_server.py`.

**State of the model (2026-10-11):** v7 meets 5 of 6 DSOs. Only DSO1 (RMSE ≤ 0.5 m/s per
level) fails, at 650–1000 m. On Colab (2026-10-12) the package reproduced v7's DSO table
**exactly** (step 0, 0.0000 m/s at every level), and 8-pass ensembling (step A) made every level
*worse* — it is rejected for DSO1. Fine-tuning (v8, steps C/D) did not fix it either:
1000 m u 0.649 / v 0.529 vs v7 0.665 / 0.508, global v worse — v8 is not better than v7, which
stays the served model. Remaining lever: a spatially varying geostrophic input (full retrain).

`notebooks/archive/` holds the architectures explored and then dropped (FNO2D variants, U-Net/FNO,
GNN, DeepONet, ResNet, ViT). They are history, not alternatives to maintain — do not "fix" them.

## Commands

```bash
# local validation, no Colab, no GPU, under a minute — run this after touching the pipeline
powershell -ExecutionPolicy Bypass -File scripts\smoke.ps1
python -m pytest -q                       # 30 tests, CPU only, mini-dataset fixtures

python -m pcfno3d info                    # effective config + which artifacts exist
python -m pcfno3d build-cache             # day CSVs -> per-instant .npy tensors (once, slow)
python -m pcfno3d check                   # geometry + ground-truth sanity
python -m pcfno3d train                   # resumes automatically from artifacts/.../last.pt
python -m pcfno3d evaluate                # test metrics + DSO1-DSO5 verdict
python -m pcfno3d reconstruct --stem 004_120000 --altitude 120 --web

pcfno3d apikey create app-web --scope read   # secret shown once; only its hash is stored
pcfno3d serve --port 8000                    # web app /app/ · API /api/docs · MCP /mcp
pcfno3d mcp                                  # same MCP tools over stdio
```

Every command takes `-c/--config` (`default`, `test`, `colab`, or a path) and
`--set section.key=value`. Config precedence: YAML < `--set` < `ANEO_DATA`/`ANEO_CACHE`/`ANEO_RUN`
env vars (the notebook's test-mode variables, kept for compatibility).

## Architecture of the package

One module per notebook block; `Config` and `Geometry` are passed explicitly instead of the
notebook's globals.

- **`config.py`** — dataclasses for every constant of BLOC 1 (`ModelCfg`, `TrainCfg`, `LossCfg`,
  `MaskCfg`, `DSOCfg`, `Paths`). Relative paths in a YAML resolve against the repo root, so
  commands work from any directory.
- **`geometry.py`** — `Geometry` holds everything instant-independent: model grid (read from
  `meta.json`, never hardcoded), terrain, `valid` / `mass_mask` / `first_valid` masks, the 6+3
  static input channels, and the discrete operators `grad1d` / `mass_residual` (same scheme as
  the generator, documented in `meta.json`). Reading the 127 MB grid CSV is cached to
  `geometry_v1.npz`, keyed by a signature — bump `GEOMETRY_CACHE_VERSION` if you change what the
  arrays mean.
- **`cache.py`** — day CSV → `{stem}_S.npy` (8, ny, nx) + `{stem}_Y.npy` (4, nz, ny, nx) +
  `index.csv`. Resumable: days already in the index are skipped.
- **`normalize.py`** — `Normalizer`; stats computed on **training days only** and written to
  `run_dir/norm_stats.json`. **That file is part of the model** — the service cannot denormalize
  without it, so it ships next to the checkpoint.
- **`features.py` / `dataset.py`** — geostrophic and thermal wind fitted by least squares on the
  *visible* sensors, 8 time/physics channels, `ANEO3D` dataset, `assemble_input` (25 channels).
- **`model.py`** — `SpectralConv3d` + `PCFNO3D`. Two deliberate fp32 islands: the FFT and the
  projection head. Removing either degrades DSO2 (bf16 rounding noise passes into the centered
  differences and floors the mass residual).
- **`losses.py`** — `PhysicsLoss`: `L_data` (w weighted per level) + `λ_grad·L_grad` +
  `φ(t)·[λ_mass·L_mass/mass_ref + λ_bc·L_bc]`. `val_score` is the model-selection criterion.
- **`train.py`** — `Trainer`: EMA weights (evaluated and saved, not the raw ones), OneCycle LR,
  gradient accumulation, atomic checkpoint writes, resume from `last.pt`. **Two-tier
  checkpoints** when `paths.local_dir` is set (both Colab configs): `last.pt` (~7 GB — model,
  EMA, Adam state) and `best.pt` go to the fast local disk every epoch, and `_sync()` copies them
  to `run_dir` (the Drive) every `save_every` epochs, at the end, and on interruption — `best.pt`
  only if it changed. Resume prefers the local copy, falls back to the Drive one in a fresh
  session. Writing straight to the Drive cost ~75 s per epoch. Don't extend `train.epochs` on a
  resume: OneCycle's schedule length is baked into the saved scheduler state.
- **The cache is never copied during training.** It doesn't change once built; `build-cache`
  mirrors only the days it just built (`cache.mirror_day`, index last). An earlier version
  re-copied the whole cache (~28 GB for v7) to the Drive every 5 epochs.
- **`evaluate.py`** — `Evaluator`: per-level metrics **accumulated over instants** (sums of
  squares, not per-instant averages — that's what a global R² requires), DSO1–DSO5 verdicts,
  sensor-robustness sweep. Writes to **`run_dir/eval_pcfno3d/<ckpt>[_ttaN]/`**, never
  `run_dir/eval_<ckpt>/`: that folder holds the *notebook's* v7 results, which are the reference
  the package is validated against. Overwriting it would destroy the comparison.
- **`dso.py`** — measure DSOs on a loader, the DSO score (worst measured/threshold ratio, same
  definition as notebook BLOC 9b), and per-level u/v weights. **Anything that tunes training
  (weights, model selection) is measured on validation, never test** — `LossCfg` refuses a weights
  file marked `"jeu": "test"`.
- **`inference.py`** — `WindFieldModel` + `Volume`: the only code path used in production. Loads
  geometry, stats and weights once; `Volume` exposes `at_altitude`, `at_point`, `to_nodes_frame`,
  `save_npz`. The checkpoint's stored architecture overrides the YAML, so an edited config can't
  silently break loading.
- **`reconstruct.py` / `export.py`** — npz + ANEO-shaped CSV + figures; `export_web` writes raw
  Float32 binaries plus `manifest.json` for `web/index.html`.

## The service layer (`service/`, `webapp/`)

Thin by design: physics and numerics belong in `pcfno3d`, never here.

- **`app.py`** — `create_app()` is a **factory, not a module-level app**: the MCP session
  manager can only be run once per instance, so every startup (uvicorn, each test) builds its
  own. `app = create_app()` at the bottom is what uvicorn imports. Routes live on a module-level
  `APIRouter` with prefix `/api`; `/app/` serves `webapp/`; `/mcp` mounts the MCP sub-app.
- **`auth.py`** — API keys as `pcf_<32>`, stored **only as SHA-256** in
  `run_dir/api_keys.json` (gitignored). Scopes: `read`, and `admin` which covers `read`.
  **No keys at all = open mode**, announced in `/api/health`. `/api/health` is the one public
  route — the web app calls it before asking for a key.
- **`mcp_server.py`** — one `MCPServer` with 8 tools over the same `WindFieldModel`.
  `set_model()` lets the HTTP app share its already-loaded model instead of loading a second
  copy. Tools in `ADMIN_TOOLS` write to disk and need the `admin` scope.
- **`webapp/`** — plain HTML/CSS/JS, no build step and no CDN. The key lives in
  `localStorage`; every call goes through `X-API-Key`.
- **`link.py`** — link mode: with an upstream configured (`pcfno3d link set`, or
  `PCFNO3D_UPSTREAM`), `create_app()` returns a relay instead of loading a model — all of
  `/api` (except `/api/health`, which describes the link) and `/mcp` are forwarded. Point
  of the design: the local URL stays stable while a Colab tunnel URL churns. **Two
  keyrings on purpose** — the client presents a *local* key, the relay injects the
  *upstream* key, so the remote key never reaches the browser. A dead upstream gives `502`
  with the repair command in the message, and `degraded` in `/api/health`.
- **`tunnel.py` / `colab.py`** — serving from Colab, where the weights and GPU actually
  are. `serve_public()` checks the weights, mints a key, opens a Cloudflare quick tunnel,
  launches uvicorn **as a subprocess** (a notebook cell dying must not leave a half-live
  server), and waits on `/api/health`. The tunnel is opened *before* the service on
  purpose — see the Host note below. Driven by
  `notebooks/current/pcfno3d_api_colab.ipynb`; `scripts/package_for_colab.py` builds the
  ~106 KB code-only archive to drop on the Drive (it refuses to ship `.pt`, `api_keys.json`
  or `.env`).

Three things that will bite if changed carelessly:

- **`link.py` imports `Request`/`Response` at module level on purpose.** With
  `from __future__ import annotations`, annotations are strings that FastAPI resolves
  against *module* globals; imported inside the function, `request: Request` is read as a
  query parameter and every relayed call returns `422`.

- **`MCPPathNormalizer`** rewrites `/mcp` → `/mcp/` *before* routing. Without it Starlette
  answers `307` on the published URL, and clients that drop the body on redirect break.
- The MCP transport enforces **DNS-rebinding protection against the `Host` header, port
  included**. `_allowed_hosts()` builds the list from `PCFNO3D_PORT` (set by `serve`) plus
  `PCFNO3D_ALLOWED_HOSTS`. Behind a proxy or domain, an undeclared host gets `421 Misdirected
  Request` — that is the first thing to check if MCP works locally but not in deployment.
  The SDK matches **exact hosts or `host:*` only** — no subdomain wildcard — which is why a
  random tunnel hostname must be known before the service starts, hence the tunnel-first
  ordering in `colab.py`.

**Fixed public URL (atom.zelqane.com, with Rayen):** `tunnel.start_named_tunnel` attaches a
connector to the *named* Cloudflare tunnel `atom` (host `atom-origin.zelqane.com`); the
credentials travel only through `TUNNEL_CRED_CONTENTS` in cloudflared's environment and are
checked (`check_credentials`) without ever being echoed. The Worker's read key is registered
**by its SHA-256 only** (`ApiKeyStore.register_hash`), so the hash can sit in
`notebooks/current/pcfno3d_atom_colab.ipynb` while the secret never touches Colab. Never
put either secret in the repo or the agents' salon.

`docs/INTEGRATION.md` is the hand-off document for whoever consumes the output: the three
interfaces, the binary format, the conventions, and the current limits. Keep it truthful — it
is what another team will build against.

## Conventions that matter

- **u east, v north, w up.** Meteorological direction is `(270 − atan2(v, u)) mod 360` — where
  the wind comes *from*. Getting this backwards is a silent 180° error.
- **Coordinates** are local metric XY (`local_metric_xy_m`); `z_terrain_m`, `z_rel_m` (above
  ground) and `z_abs_m` (ASL) are tracked separately. Model levels are ASL.
- **Values are 0 below the terrain** in cached targets, and `NaN` in reconstructed volumes
  (`mask_nan=True`). Never average over a field without the `valid` mask — the zeros would skew
  every metric.
- **`-9999.0`** is the nodata value in web exports (JSON and Float32 binaries can't carry NaN).
- DSO metrics are computed on **synoptic instants** (whole hour, multiple of 3 h); fast episodes
  (1 min / 10 s sampling) are a stress test reported separately, outside DSO1.
- Levels covering less than `dso.min_level_cover` (5 %) of the domain are excluded from per-level
  DSO verdicts — with Tunisian terrain the 25 m level is almost entirely underground.

## Data and artifacts

- `aneo_dataset_3d_v2/` (~35 GB, 100 days, 1000 instants) is the active dataset; `meta.json` and
  `split_days.json` define the geometry and the train/val/test day split. QC of the generator run
  is in `qc/qc_summary.json`.
- `aneo_physical_db_multitime_v3_progress/` is the synthetic data generator (and its MySQL /
  PostGIS schemas). `data_full/`, `data_claude*/` and the `.rar` archives are superseded datasets.
- Everything generated lives under `artifacts/` and is gitignored, as are the datasets. Do not
  regenerate or re-extract them unless the task requires it — a full generator run is multi-hour
  and multi-GB.
- `tests/` never touches the real dataset: `scripts/make_mini_dataset.py` synthesizes a 40×20
  grid, 4 levels, 6 days in the exact generator format.
