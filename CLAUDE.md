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

**State of the project (2026-10-16):** closure lots M0, M3, M7 are merged on `main`; M2, M4,
M5, M6, M8 are done and reviewed, on `agent/ines/*` branches, integrated on
`agent/ines/finalisation` — their merge to `main` waits for a PR (`gh` not logged in).
Deployment is **local only** (ines-pc: dashboard Worker under `wrangler dev` with a local R2,
v7 service on CPU, 8 test instants published). GPU verdicts measured on Colab L4
(15-16/10/2026, `docs/BILAN_FINAL.md`): DSO6 non-conforming on first reading, **conforming
on a second (non-independent) reading** (q = 0.90 chosen on validation); DSO4 **0.440 s**
for the served value with deferred reliability (reliability ready 3.8 s later, no verdict);
AN1 v1 non-conforming, **v2 conforming on a second reading** (drifts 100 %, false alerts
2.2 %); EOL1 1.71 % on the **demonstration** farm only (official verdict needs the real
farm); M6 replayed the 120 test instants; M8 measured (L6: max u, v RMSE v7 0.665 vs 5.027–6.487
m/s; E2 is a negative result). **M1 delivered** (cell 10i, Colab A100, 2026-10-16,
`docs/M1_PROTOCOLE.md`): the post-model correction passes C1 (worst level u 0.419, v 0.382)
and C3, but **fails C2** on the 46 episode instants — **not retained**. **v7 is frozen
alone** (annotated tag `v7-fige`, `docs/resultats/modele_fige/README.md`): the tag is the
**reference** every later result is compared to, **not a ban on retraining** — Ines trains
and builds the models and decides retrains herself (2026-10-17). A retrained or new model
gets its own name and `run_dir` (never overwrite `runs/v7` or the `v7-fige` results) and is
reported against v7. DSO1 is still
not met at 650–1000 m and is reported with the WMO criterion. The correction stays on
`agent/ines/m1-correction-post-modele`, unmerged. Details: `tukhnanutha.md` § 5 "Delivery
state".

**v9 — the single model going forward (decision of Rayen and Ines, 2026-10-21).** The v7
method (same FNO 3D, physics loss, targets u, v, w, ρ from 5 m to 1 km above ground) trained
on **real data over all of Tunisia**: inputs = airport anemometers (METAR, horizontal wind at
10 m, w channel = 0), training truth = ERA5-Land surface (0.1°) + ERA5 aloft, grid 23 × 78 × 45
(≈ 9 × 11 km), extra validation on real IGRA radiosondes (Tunis-Carthage, Tozeur). Configs
`v9_colab.yaml` (training, `notebooks/current/pcfno3d_v9_colab.ipynb`, Colab A100) and
`v9.yaml` (ines-pc, serving); data build `scripts/tunisie_fno3d.py --grille land`
(`pcfno3d.tunisie_fno3d`). v9 is the model to serve on atom.zelqane.com once trained and
scored; the synthetic v7 stays as the comparison reference and proof of method (100 m detail
has no real-data equivalent). The name v8 is taken (the rejected 2026-10-12 fine-tune).
ERA5 is a reanalysis: say "training truth", never "measurements". Forecasting (E19, 10–30 min)
is a future extension: v9 is a reconstruction/analysis of the current instant.

**PC-FNO3D réel (Grand Tunis) — THE deployed model (decision of Ines, 2026-10-21, after
v9 above).** v7 is its base (same operators, starts from v7 weights, `configs/grand_tunis.yaml`
= v7 architecture), v9 its insurance. Domain 68 × 56 km (36.395–36.900 N, 10.04–10.80 E),
surface 50 m (Copernicus DEM + ESA WorldCover, `scripts/grille_reelle.py`), nodes 100 m,
v7 levels. **Training truth = the generator forced by ERA5** (`scripts/forcage_era5.py` →
`forcage_era5.csv/.npz`; `generate_aneo_3d.py --forcage`: `DayStateERA5`, sea breeze from
the land–sea contrast, large scale recalibrated on ERA5 cell by cell at the same height
above ground, w by continuity). **Stations never enter the truth** — only the input, fine-tuning
outside test blocks, and validation (rule of Ines). Stations in the domain: Tunis-Carthage
(METAR), Zaghouan 60729 and Nabeul 60728 (SYNOP via NOAA ISD, `scripts/isd_synop.py`; ends
2025-08-24). Split: test = Jan–Mar 2022 + Jul–Aug 2022, val = Jan–Feb 2021 + Jul 2021,
train = the rest with 7 days of margin. `--cache-dir` writes the model cache directly
(`_S` float32 pooled 2×2, `_Y` float16, `_E` = ERA5 10 m wind, `index.csv` with ERA5 ug/vg/shear;
~49 MB/instant) plus `data_dir/stations.json`. Input mode `mask.entree: era5_stations` =
ERA5 10 m pseudo-sensors every 2.5 km + the station cells (the production input; v7's dense
layer would leave the model blind with 3 stations). Gates before full generation:
`scripts/test_generateur_reel.py` (winter J2: mass exact, domain mean ≈ ERA5, all three
stations ≤ ERA5 RMSE — passed 2026-10-21) and `scripts/brise_mer.py` (summer diurnal
direction cycle at Tunis-Carthage and Nabeul, figure shown to Ines before generating).

**Grand Tunis v2 (2026-10-22).** v1 (`runs/grand_tunis`, stopped at epoch 14/40) plateaued at
R² u 0.78 / v 0.73; `scripts/diag_grand_tunis.py` on the 64 validation instants showed ERA5
interpolated to the nodes alone gives 0.95 / 0.97 (alignment checks all OK, generator noise
≤ 3 % of spatial variance): an input gap, not a bug. `paths.forcage_era5` (`pcfno3d.forcage`)
adds 7 forcing scalars (ug, vg, shear u/v, veer, stability, land–sea ΔT) and u, v ERA5 at every
node, computed exactly like the generator (`_era5_au_sol`, checked to 1e-5 m/s on the real
grid) — 34 input channels, the new ones LAST. The ERA5 cells (2 × 14 × 4 × 4) ride in `tfeat`
after the 8 time channels, so loaders keep their shape; `assemble_input` densifies them on the
device. `model.residuel_era5`: u, v = ERA5 (last two input channels) + learned correction.
`init_from` a 25-channel checkpoint keeps the old lift columns, zeroes the new ones and (for a
residual model) the u, v rows of the last projection, so training starts at ERA5. Serving needs
the instant's `timestamp`. Two modes: `grand_tunis_v2_colab.yaml` (real: ERA5 10 m + stations,
from v1 `best.pt`) and `grand_tunis_v2_aneo_colab.yaml` (simulated dense ANEO network, from v7);
notebooks from `scripts/make_grand_tunis_nb.py`. **v2 real-mode result (Colab A100, 2026-10-22,
20 epochs, `runs/grand_tunis_v2/eval_pcfno3d/best`)**: validation R² u 0.986 / v 0.992 / w 0.908
(v1 0.781 / 0.729 / 0.609); test, 180 synoptic instants: global RMSE u 0.478 / v 0.453 m/s, R² u, v
≥ 0.975 at every level, direction MAE 3.9°, DSO4 290 ms. Not met: per-level RMSE ≤ 0.5 (0.48–0.59
below 75 m and at 400–500 m), DSO2 mass 1.77e-4 (> 1e-4: ERA5 added in the residual is not
divergence-free), DSO3 roughness 0.76 (too smooth). The DSO5 sweep is meaningless in
`era5_stations` mode (`keep` is ignored: identical rows) — robustness there means dropping
stations / ERA5 points, still to write. Validation against real stations and radiosondes still to
do. **Night of 22→23/10 (same Colab session, test = 180 synoptic instants):**
`grand_tunis_v2_long` (v2 + 40 epochs at LR 1e-4, `runs/grand_tunis_v2_long`) is the best model:
global RMSE u 0.377 / v 0.371 m/s, per-level RMSE ≤ 0.42 from 75 to 1000 m and 0.53 / 0.51 at 25 m
(only miss of the per-level 0.5 criterion), R² u, v ≥ 0.98, direction MAE 3.3°, DSO2 1.31e-4
(still > 1e-4), DSO3 0.82. `grand_tunis_v2_aneo` (simulated dense ANEO network, from v7, 20
epochs) is WORSE than the real mode: global 0.548 / 0.514, per-level up to 0.67, DSO2 2.33e-4 —
the dense synthetic surface layer does not beat ERA5 profile + 3 stations once the ERA5 profile
is an input. Drive storage reached 91 % (each run keeps a ~7 GB `last.pt`). **Validation on REAL
measurements (2026-10-23, `scripts/valider_grand_tunis_reel.py`, v2_long, 180 test instants, real
anemometers as input, each station left out when judged, 10 m wind):** vector RMSE model / ERA5 10 m /
generator surface = Nabeul 2.42 / 2.57 / 2.52, Tunis-Carthage 2.70 / 2.81 / 2.63, Zaghouan 2.28 /
2.16 / 2.14 m/s; direction MAE 27–38° for all three sources. At the stations the model is at ERA5's
level (a few % better at two of three): it reproduces the generator, whose own error against the
stations is the same — the limit is the training truth, not the network. Radiosondes Tunis-Carthage
(only 9–11 soundings per 250 m layer on the test blocks): model better than ERA5 in every layer
0–1000 m (vector RMSE 1.95–2.34 vs 2.07–2.54 m/s) with a smaller speed bias near the ground (0.58 vs
1.04 m/s below 250 m). **v3 = fine-tuning on the real stations (2026-10-23/24,
`pcfno3d.stations_reelles`, `loss.lambda_station` 0.5, 10 epochs from v2_long):** the final model
(all 3 stations as targets) looked spectacular at Tunis-Carthage 1.99 and Zaghouan 1.74 m/s (v2
2.70 / 2.28) but worse at Nabeul 2.70 (bias −1.47: the sheltered station reads too little wind and
the model followed it). The **spatial cross-validation** decides: with the judged station NEVER a
target, Tunis-Carthage 2.66, Zaghouan 2.27, Nabeul 2.41 vs v2 2.70 / 2.28 / 2.42 — −1.5 / −0.4 /
−0.5 %, inside the noise (~170 instants per station). The v3 gain is a **site correction** learned
at the target locations, not a better field between stations. Delivery rule (Ines, 2026-10-23,
`scripts/bilan_plis_grand_tunis.py`: ≥ 5 % mean gain and ≥ 3 % at 2 of 3 never-seen stations):
**v2_long stays the delivered model**. v3b (Nabeul direction-only, `loss.poids_vitesse_station`),
trained on request, is discarded too: Nabeul 2.79, Tunis-Carthage 1.99, Zaghouan 1.75 m/s — the
same site correction. **Final decision (Ines, 2026-10-24): end of training; the delivered model is
v2_long, copied to `runs/grand_tunis_v9_production`** (best.pt, norm_stats, evaluations, real
validation; done by the serving notebook). v3/fold/v3b results and the comparison maps are kept for
the report; their best.pt are deleted except the fold without Nabeul. Nabeul 2023–2024 measurements
are excluded (`stations_reelles.EXCLUSIONS`, on top of the detected outage months). Next: plug the
ATOM Wind front-end on the API with v2, then the OS6 plan on Grand Tunis. The 3D field is unharmed by the fine-tuning (generated test identical to v2).
**v4 (DSO fine-tuning, accepted by Ines 2026-10-24 with rules):** `configs/grand_tunis_v4_dso_colab.yaml`,
`notebooks/current/pcfno3d_grand_tunis_v4_dso_colab.ipynb` (not run yet: Colab only on "start
execution"). w is recomputed by continuity (`inference.w_continuite`, `Geometry.w_continuite`) in
the served pass, the evaluation, `dso.measure` AND every masked-ensemble pass (`_members`, so DSO6
judges the served w); DSO2 with it is reported as "guaranteed by construction" (measured value
shown, with the truth density). Evaluation tables now carry the per-level DSO3 (`rugosite`). The
test is read once (steps skip when their outputs exist); the old v2 evaluation is kept as
`eval_pcfno3d/best*_avant_v4`. `scripts/decision_v4_grand_tunis.py` (rule fixed before the v4 test:
v4 delivered only if no worse than v2 on DSO1, DSO4, DSO5, real stations/radiosondes, 1 % RMSE
tolerance, no station > 3 % worse; else v2 with w by continuity; exit 0 / 3 / 2) writes
`runs/decision_v4.json`, `comparaison_v2_v4_par_niveau.csv`, `rapport_v4.md`. DSO6 (masking ensemble,
calibration on validation, 80 % coverage + Spearman on test) then runs on the retained model without
retraining. After that, no more training. The serving notebook copies the model named by
`decision_v4.json` (default v2_long) to v9_production, serves with `w_continuite: true`, and asks for
an L4 GPU (A100 if unavailable or out of memory).
**v4 result (read on the Drive 2026-10-30, `runs/decision_v4.json` of 24/10): v4 DELIVERED**
(`modele_livre: grand_tunis_v4_dso`, w by continuity) — no worse than v2 within tolerance: global RMSE
u 0.377 → 0.380, v 0.371 → 0.373 m/s, worst level 0.530 → 0.528, DSO4 0.302 s both, DSO5 (25 %)
0.537 → 0.535, real stations mean 2.467 → 2.469 m/s (Nabeul 2.42, Tunis-Carthage 2.70, Zaghouan
2.28 / 2.29). DSO2 with w by continuity 1.05e-5 (without: v2 1.31e-4, v4 1.25e-4). DSO6 on v4
CONFORMING: 80 % coverage u 0.878, v 0.883, w 0.841, speed 0.879; Spearman u 0.16, v 0.17, w 0.50,
speed 0.19 (180 synoptic instants). The serving notebook therefore serves v4.
Proposed DSO variants (Ines, 2026-10-24; reported, NOT used for the decision until the
supervisor validates them): DSO3 against the predictable part of the truth
(`scripts/dso_variantes_grand_tunis.py`: test instants regenerated with the generator's A_ROT/A_DIV
perturbations off, same forcing and seed, via `scripts/diag_grand_tunis.py`; replicas with the
original seed check the regeneration), and DSO1 on u, v R² only with w reported apart.
**ATOM Wind front-end (2026-10-25).** Ines's mockup (`docs/atom_wind/atom_wind_maquette.html`, one
HTML file whose wind came from an in-page sea-breeze formula) is the SOURCE; `scripts/build_atom_wind.py`
adds anchors (fails if one is missing) and writes `webapp/atom-wind/index.html`, served at
`/app/atom-wind/`. `webapp/atom-wind/atom-data.js` loads, before the page script, `/api/atom-wind/grille`
(real terrain, water from z0 < 0.001, z0, stations on the page's 500 m grid), `/jours` (test instants
grouped by Tunis local day, UTC+1), `/jour?date=` (u, v at AGL heights 10–1000 m, 500 m blocks, int16
cm/s base64, gzip; + σ speed if `inference.reliability` is set) and `/reseau-aneo` (OS6 GeoJSON);
hours between model outputs are interpolated linearly. `pcfno3d.atom_wind.champ_agl` is shared with
`scripts/comparer_modeles_grand_tunis.py`. Without the API the page keeps its example day and says so;
a 401 shows a key field (localStorage `pcfno3d.apikey`, never in the URL). The reliability view says
whether σ is the calibrated model σ or the page's distance heuristic, and whether the ANEO network is
the OS6 plan or the page's example layout. The real-validation table in the page is static (v2 values).
**Cahier v12 (2026-10-30, Downloads/Cahier_des_charges_ATOM_Wind_PC-FNO3D_v12.pdf, confidential)** adds OS7 (3D wind forecast at +24/+48 h: the v2 model fed by GFS instead of ERA5, fine-tuned on archived GFS forecasts 2021–2025 with the same truth/split, judged on 2022 vs raw GFS and persistence: PREV1–PREV3) and OS8 (agriculture = main app 2–10 m, J0–J+2 treatment/irrigation windows, FAO-56 water need with T/RH/radiation from GFS, AGRI1/AGRI2 vs INM stations 2022; wind energy = 3-function demo 50–150 m, EOL2 radiosondes), front-end section 17 (V3D1–V3D10, TB1–TB7, last real state with timestamp when the service is down). The real-time source is therefore GFS (Open-Meteo `gfs_global`, all 11 ERA5 pressure levels) instead of ECMWF IFS. Checked 2026-10-30: Open-Meteo's Previous Runs API has GFS WIND at +24/+48 h only recently (2025-06 yes; 2022-01, 2022-07, 2023-03, 2024-01 all null; pressure levels have no previous_day at all), so the OS7 archive must come from NOAA's GFS bucket on AWS (noaa-gfs-bdp-pds, GRIB2 0.25°, byte ranges via .idx). The new mockup (Leaflet/OSM basemaps, J0–J+2 agriculture, no Nesma/chantier) is the builder source; `/atom-wind/direct` also serves t2m/hr2m (GFS, 500 m blocks) and domain-mean shortwave radiation; the page's windD/Tair/RHair/et0Day use J0–J+2 when real time is loaded. The Worker keeps every successful `/api/atom-wind/*` GET in its cache and serves it with `x-atom-dernier-etat` when the Colab is down (page banner « Service hors ligne : dernier état réel connu »). On ines-pc, stale `workerd` processes kept port 8787 and swallowed requests: kill every wrangler/workerd before `wrangler dev`.
**v12 additions (2026-10-30)** — `pcfno3d.gfs` + `configs/grand_tunis_v5_gfs_colab.yaml` + notebook `pcfno3d_grand_tunis_os7_colab.ipynb` (OS7: archived GFS f24/f48 from NOAA AWS, 9 pgrb2.0p25 levels, `MaskCfg.e_source: forcage`; PREV1/2 `scripts/evaluer_prevision.py`, PREV3/AGRI1/AGRI2 `scripts/verifier_decisions.py` with the treatment/irrigation rule frozen from the mockup's ACTS). GEFS (31 members, 10 m only) gives `/api/atom-wind/ensemble?j=` (probability of a treatment window). Atlas E8/E15: `pcfno3d.atlas` + `pcfno3d.sig` (dependency-free GeoTIFF), `/api/atom-wind/atlas`. Assistant E6: `service/assistant.py` (Claude via `ANTHROPIC_API_KEY`, tools from `service/atom_outils.py`, 503 without key), `POST /api/assistant`. SMS E12: `service/sms.py` (journal by default, Twilio with `PCFNO3D_SMS=twilio` + `TWILIO_*`; admin routes `/api/sms/*`, numbers masked). 3D view V3D1–V3D10: `webapp/atom-wind/atom-3d.js`, called by `ATOM.vue3d(ctx)` at the end of the mockup's `init3d` (builder anchors 13); it replaces the particle loop, exposes exaggeration ×1–×10 (default ×5), AGL (120 m default) / altitude, streamlines, horizontal cut, colour by speed / uncertainty / reliability, layers, timeline, point sheet with 80 % profile band, and lowers pixel ratio / particle count below 30 fps. Trilingual UI (E17/TB7): `webapp/atom-wind/atom-i18n.js` translates the rendered page (text nodes + aria-label/title/placeholder; `{n}` patterns whose captured values are translated too; Arabic RTL; `localStorage` key `aw-langue`, switching reloads) from `i18n/ar.json` / `i18n/en.json`; `scripts/extraire_textes_atom_wind.py` (a small JS literal scanner over the page script, `atom-data.js` and `atom-3d.js`, plus `i18n/textes_supplementaires.json` collected at runtime via `ATOM.nonTraduits`) writes `i18n/textes_fr.json`, and `tests/test_atom_wind.py` fails if it is stale or a dictionary drops under 90 %. Code that READS displayed text must use `ATOM.texteFr(el)` (the French original). `atom-data.js` extends `window.ATOM` (atom-i18n.js creates it first) — never reassign it. TB5/TB1: `GET /api/atom-wind/qualite` (`atom_wind.qualite`) reads the SERVED run_dir (eval best_wcont, `dso6*.json`, real stations, `decision_v4.json`); the dashboard shows it instead of the mockup's frozen v2 numbers, plus active stations / reconstruction time / hub wind / mass residual. `scripts/bilan_conformite.py --runs <runs>` (run at the end of the OS6 and OS7 notebooks) classifies every v12 acceptance criterion from written results only (REEL1 adds an IDW of the OTHER stations computed from `stations_paires.csv`; EOL2 = radiosondes 0–250 m vs ERA5). Gap closure vs v12 BF list (2026-10-31, gap analysis in git-ignored `artifacts/cdc/ecarts_bf.md`): `/api/atom-wind/agricole` and `/production` (S16); per-instant mass residual / ground condition in `jour_direct` (BF11/BF31) and optional real-time sigma with `PCFNO3D_SIGMA_DIRECT=1` (BF8); on-disk history `run_dir/historique/*.json.gz` (every real-time day and each J+1/J+2 forecast with its issue date; `/api/atom-wind/historique[/{nom}]`, MCP `historique_atom_wind`, page "Historique" group; `PCFNO3D_HISTORIQUE=0` disables; ~8 MB/day each); real-time service alerts (silent station by cadence, mass residual, strong wind; page via anchor 17, server `/api/atom-wind/alertes`) and one-time notifications (`service/notifications.py`: journal always, SMTP `PCFNO3D_SMTP_*` + `PCFNO3D_NOTIF_EMAILS`, WhatsApp via Twilio `TWILIO_WHATSAPP_DE` + `PCFNO3D_NOTIF_WHATSAPP`; thread with `PCFNO3D_NOTIFICATIONS=1`); PRODUCTION input for the real model on `/reconstruct/stations`, CSV upload and `publish --csv` (`temps_reel.entree_production`: GFS of that hour via Open-Meteo, ~3 months back to 16 days ahead, each measurement at 10 m in its cell; 503 if GFS is unreachable) — before, these fed the synthetic dense layer; cache manifest (`cache_manifest.json`, `pcfno3d manifeste-cache [--verifier]`, checked by `info`); `/api/exports/{stem}.nc|.csv`; admin key management `POST/PATCH/DELETE /api/keys` + admin-page card; `pcfno3d surveiller <dir>` (validate, reconstruct, publish, export, then move to traites/ or quarantaine/ — never deletes); `/api/supervision` (admin). Then (same day): Jensen wake for the virtual park (`pcfno3d.sillage`, `/atom-wind/variations` gives power with and without wakes, BF72); outage scenario (`atom_wind.scenario_panne`, `/atom-wind/scenario`, MCP `simuler_panne`, BF88); OS6 plan resumed from installed sensors (`--installes`, Drive `os6_installes.csv`, BF78); GFS gust factor in real time (`wind_gusts_10m`, `facteur_rafale`, page anchor 18 replaces `gustF`); per-node wind columns layer in 3D (BF32); virtual-station value in silent-station alerts (BF37); `docs/MODEL_CARD_GRAND_TUNIS.md` (delivered v4) and the ATOM Wind routes table in `docs/INTEGRATION.md`. Perspective spaces (v12 §6.11, `service/atom_outils.py`, statistics over the days at hand — J0–J+2 plus the real-time history): Porteur mast sites `/atom-wind/porteurs` (BF85, 15/20/25/30 m, generic small-turbine curve scaled, OS6 plan sensors ranked too), crane wind/gust at jib height `/atom-wind/grue` (BF86, gust = GFS 10 m factor decayed with height like the page's gustF; counterweight left to the maker's charts), windbreak hedges `/atom-wind/haies` (BF87, direction of winds above a threshold weighted v²(v−s), hedge perpendicular, sheltered zone 8 H / 15 H leeward, 3 H windward, GeoJSON). MCP has 22 tools. Accounts (BF19/BF43): `service/comptes.py` (scrypt-salted passwords in `run_dir/comptes.json`, session = an ApiKeyStore key named `compte-<sha12(email)>`, one per account, 5 failures lock 15 min, deactivation revokes at once); `require_eolien` restricts the wind-energy routes for SESSION keys only (service keys such as the Worker's keep access); `pcfno3d compte-admin EMAIL` (password from `PCFNO3D_MOT_DE_PASSE_ADMIN`); the page's login/register/logout use the server when it answers (anchor 20) and send the session in `X-ATOM-Session`, which the Worker forwards as `X-API-Key` (otherwise it injects its read key). The serving notebook moves an older production copy aside (`grand_tunis_v9_production_avant_<new>_<old>`, renamed, never deleted) when `decision_v4.json` names another model.
**Real time and forecast (2026-10-26)** — `pcfno3d.temps_reel` + `atom_wind.jour_direct` + `GET /api/atom-wind/direct?j=0|1|2` (today in real time, tomorrow, day after): ERA5 is ~5 days late, so the large scale comes from ECMWF IFS 0.25° via Open-Meteo (no key) on the SAME cells as the training forcing (`geom.forcage.lat/lon`) with the SAME formulas as `scripts/forcage_era5.py` (levels 1000/925/850/700 hPa only, cell orography = Open-Meteo DEM elevation, land-sea dT = t2m land − sea, the no-slx rule); `ForcageDirect` has the ForcageERA5 interface and `predict(..., forcage=vec)` takes it. Stations: METAR DTTA (aviationweather.gov) + SYNOP 60729/60728 (OGIMET, `lire_synop`), nearest within 30 min, past hours only; future hours = forecast without stations. Surface T/p/RH/rho from IFS brought to the grid relief. Cached 30 min per day in the service; the page defaults to « Aujourd'hui, temps réel », sets the slider to the Tunis hour and refreshes every 30 min. TLS uses the system store (truststore, else certifi): ines-pc's Python CA bundle is outdated. Real time needs the service running (Colab GPU); a Colab session is not 24/7 hosting.
**OS6 RESULT (run 2026-10-31, v4): no gain from added sensors** — every network of 10–200 sensors (grid, random, optimised) stays at 0.5326 m/s vector RMSE / 0.528 worst level on test, `nombre_recommande: null`; the v4 net uses the ERA5 lattice and treats point measurements as local site corrections. Sizing the network needs a sensor-aware model (retraining = Ines's call); details in `tukhnanutha.md` § 5. **OS6 (2026-10-25)** — `pcfno3d.os6` + `scripts/os6_reseau_aneo.py` + notebook
`pcfno3d_grand_tunis_os6_colab.ipynb` (L4, delivered model from `decision_v4.json`, no training): OSSE
where an ANEO sensor is a cell whose cache surface truth (10 m mast on the real grid, noise included)
joins the production input like a station. Candidates (land z0 ≥ 0.005, ≤ 650 m, slope ≤ 20°, 1 km
from border/stations/each other, optional lat/lon exclusion polygons in `os6_contraintes.json` — none
invented); greedy placement on VALIDATION (5 per step at the max of the smoothed residual error,
re-run, gain per step); sizing curve on TEST once (optimised vs regular grid vs random ×3, 10–200
sensors; recommended = smallest N with worst-level RMSE ≤ dso.rmse); plan S18 CSV/GeoJSON/map. Limit:
the delivered model was trained with ERA5 + 3 stations, so the gain is that of the CURRENT model.
Maps v3 − v2 (`scripts/comparer_modeles_grand_tunis.py`, 30 test instants, `runs/comparaisons/`)
show the site correction as **spots around the stations**: mean vector difference at 10 m AGL
0.82 / 0.66 m/s within 1 km of Tunis-Carthage / Zaghouan, 0.3–0.4 at 1–3 km, 0.12 at 3–10 km,
0.07 beyond (domain 0.075); speed lowered by 0.37 m/s near those two and 0.97 m/s near Nabeul; at
100 m AGL only 0.15–0.26 m/s near the stations (domain 0.057).
**Serving the delivered model (2026-10-24):** `configs/grand_tunis_v2_service_colab.yaml` + `notebooks/current/pcfno3d_grand_tunis_atom_colab.ipynb` serve v2_long on atom.zelqane.com (same named tunnel / key fingerprint as the v7 notebook). In `era5_stations` mode `WindFieldModel.surface_from_cache` now builds the PRODUCTION input (ERA5 10 m lattice from `_E` + stations, real METAR/SYNOP if `paths.stations_obs`) and puts that mask in `S[7]` — before, the service fed the generated dense surface layer, an input the real model never saw. CSV uploads and `/api/reconstruct/stations` still lack the ERA5 lattice for this model (cached test instants only). Also fixed that day: METAR u, v now filled on
import for every non-calm report (`metar.py`), IGRA launch time of 00 UTC balloons dated the
previous day (`igra_radiosondages.py`; matching was already on `temps`).

**OS7 RESULT (A100, 2026-11-07; v5 = delivered v4 fine-tuned 12 epochs on GFS f24, results in
`<Drive>/pcfno3d_grand_tunis/os7/`; 991/996 of 1000 GFS f24/f48 instants, the 9 missing NOAA files are
dropped by `load_index`):** PREV1 (+24 h) CONFORMING, vector RMSE at the 3 stations v5 2.64 vs raw GFS 2.89
vs persistence 4.09 m/s, better than GFS at every station; PREV2 (+48 h) CONFORMING, 2.82 vs 3.09 vs 4.51.
Reference (delivered model fed by GFS, no fine-tune) 2.81 / 2.99, worse than GFS at Zaghouan — the fine-tune
is what makes it conform. PREV3 (treatment decision the day before vs measured): agreement +24 h 65.6 %
(κ 0.36) vs GFS 63.1 % vs persistence 60.7 %; +48 h 64.6 % vs 61.5 % vs 57.8 %; dangerous "yes" 2.75 % /
1.96 % (≤ 5 %, but raw GFS is lower, 1.2 % / 1.0 %). AGRI1 (delivered model, station hidden) 62.7 %
agreement, AGRI2 2.36 % conforming. `bilan_conformite.py` now also reads `<runs>/../os7/` (it reported PREV
"à mesurer" before). **v6 DSO (2026-11-07, rule fixed before the test): NOT delivered, v4 stays** — the
roughness term reached DSO3 (0.939, conforming) but DSO1 got worse (worst level 0.693 vs 0.528, 25 m),
DSO5 0.696 (> v4 + 1 %) and the real stations 2.544 vs 2.469 m/s: the DSO1/DSO3 trade-off predicted
before the run, now measured. v4's own DSO3 is 0.805 (0.37 at 25 m … 0.93 at 650 m).

**National model — state 2026-11-09.** v2 TRAINED (Colab A100, 30 epochs, no overfitting,
`MyDrive/pcfno3d_national/runs/national_1km_v2/best.pt`) and delivered as the national reference; test 2022:
DSO1 non-conforming (worst level 1.158 m/s at 2500 m; 0–1 km AGL 0.881), DSO2/DSO4/DSO6 conforming, DSO3 0.785,
DSO5 1.136, airports 3.045 vs ERA5 2.857 m/s. v3 (roughness fine-tune) rejected by `configs/national/regle_v3_rugosite.json`.
Diagnosis (`docs/diagnostic_v_national.md`): v is 1.4× worse than u because of the TRAINING TRUTH — the generator's
w-damping (potential flow added after the ERA5 recalibration) explains 85–92 % of truth−ERA5 at 300/800 m AGL; fix
written (`ANEO_GEN_AMORT_PASSE_HAUT=1`, not used for any generation yet). Ines's national roadmap (phases 1–9,
`docs/feuille_de_route_nationale.md`): DSOs judged 0–1 km AGL (`scripts/dso_hauteur_sol.py`); every computation has a
dated, hashed rule written first (`configs/national/regle_*.json` + `.sha256`, verdicts by `scripts/juger_phase1.py`);
tuning on validation only; test 2022 read once per model; NO training or full generation without her written
approval, phase by phase; no noise added to predictions; each new model its own run_dir. Grand Tunis is frozen (v4,
no more training). The Colab notebook `pcfno3d_national_colab.ipynb` no longer trains (section 4 = phase-1
diagnostics). **Seed mean of C' (v4, PROPOSED rule `configs/national/regle_ensemble_cp.json`, not run):**
`pcfno3d.moyenne.ModeleMoyen` loads N run_dirs (each its own best.pt + norm_stats, one shared `Geometry` via
`WindFieldModel(geom=)`), serves mean u, v with w by continuity of the mean (`predict`, sequential passes keeping
only sums; `benchmark` = DSO4 like `WindFieldModel.benchmark`); `predicteurs()` plugs the mean and each seed
(cached passes) into `metriques_loader` / `dso_hauteur_sol.mesurer(..., jeu="validation", predicteurs=)`.
`scripts/ensemble_graines.py` (validation only: bands 0–1 km, DSO5, DSO4, inter-seed spread calibrated on
validation with method `ensemble-graines-v1` — `WindFieldModel` refuses it for served reliability) →
`runs/national_1km_v4_Cp_ens/`; notebook `scripts/make_ensemble_nb.py`. Test of the mean: separate approval.
History of the start:
**National model (started 2026-11-06, Ines's request).** The Grand Tunis v4 method over
all of Tunisia: `configs/national_1km_colab.yaml`, domain 30.15–37.615 N × 7.45–11.644 E (828 × 388 km),
surface 500 m (1656 × 776), nodes 1 km, 16 ASL levels 25…2500 m (`generate_aneo_3d.py --dx 500 --niveaux …`,
`forcage_era5.py --niveaux …`; defaults unchanged for Grand Tunis), inputs = ERA5 10 m every 2 km + the
12 METAR airports (`configs/national/stations_metar.csv`, `grille_reelle.py --stations-csv`; stations carry
their `code` into `stations.json`). Grid from 46 Copernicus/WorldCover tiles (`grille_reelle.py` now reads
a WorldCover mosaic tile-by-tile; the Grand Tunis grid rebuilt with it is byte-identical). Forcing from
`AppData/Local/pcfno3d/era5/tunisie` (the COMPLETE ERA5: `tunisie_donnees.zip` lacks 2021-07 and
2022-07/08, i.e. validation/test months). Documented approximations: constant-step local plane (E–W scale
1.041 south / 0.954 north), constant Coriolis 8.13e-5 (7.3–8.9e-5), country-mean forcing scalars (cell-by-cell
ERA5 recalibration carries the north–south differences). Notebooks `scripts/make_national_nb.py`
(generation CPU 3 parts ≈ 50 GB on the Drive under `pcfno3d_national/`, then training A100 from the
delivered Grand Tunis weights); data `scripts/package_national_colab.py` → `national_donnees.zip`.
Still to write: national real validation (`stations_reelles` / `valider_grand_tunis_reel.py` are Grand
Tunis-specific). The OS7 GFS download (`pcfno3d.gfs`) also keeps the national box (`*_tn.npz`,
`gfs_tunisie_f<L>.npz`) for a later national forecast.

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
python -m pcfno3d validate f.csv --quarantine quarantine/   # data contract, exit 0/1/2
python -m pcfno3d train                   # resumes automatically from artifacts/.../last.pt
python -m pcfno3d evaluate                # test metrics + DSO1-DSO5 verdict
python -m pcfno3d reconstruct --stem 004_120000 --altitude 120 --web
python -m pcfno3d alertes --split test --limit 8   # M4 alerts, chronological, JSONL journal
python -m pcfno3d publish --stem 004_120000 --exports   # M6: ATOM rev. 2 snapshot, local
python scripts/m6_rejeu_colab.py -c colab            # M6: 120 test instants, as live

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
- **`contract.py`** — the ANEO data contract (spec §8.1–8.3, criterion DE1): two Pandera
  schemas validated `lazy=True` (raw: columns/types; typed: ranges + §8.2 rules), §8.3
  quality indicators, verdict `conforme` / `quarantaine` / `rejet` with the thresholds as
  constants at the top of the module, and the quarantine move/copy (never deletes).
  `pcfno3d validate` and `POST /api/reconstruct/upload` use it (`build-cache` does
  NOT). Upload uses `Seuils.upload()`: absent / `mask_i_obs=0` / no-u-v-T stations are a
  known outage (DSO5), only a `rejet` gives 422; quarantine still reconstructs, with
  `statut_qualite`. Stations the contract flags (out of range, frozen, NaN) are masked
  before inference (`predict_csv(exclure=...)`; NaN is always masked there). Spec,
  thresholds and spec-vs-files gaps: `docs/CONTRAT_ANEO.md`; fixtures in
  `tests/data/contract/` (regenerate with `scripts/make_contract_files.py`).
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
- **Deferred reliability (M2, 2026-10-16):** DSO4 times the served value only; the ensemble
  runs apart (`assess_reliability[_async]`, `Volume.reliability_state`) — `docs/FIABILITE.md`.
- **`alerts.py` / `an1.py`** (lot M4, criterion AN1) — one alerts document per instant in the
  ATOM rev. 2 format (`alerts.schema.json`: id = sha8 of `kind|target`, 4 levels, cause and
  recommendation), aggregating M3 (contract report), M2 (coverage), M4 (sensor drift — v2
  default `spatiale`: measurement vs median of its neighbours minus a per-cell site offset
  learned on TRAIN days only (`ModeleSites`, `sites_R<r>.npz`), vector persistence, no model
  pass; v1 `innovation`/`influence` by grouped leave-out, `derive_cote`² extra passes —
  input drift on T/rho/RH/p only (wind excluded), mass residual / D11 `physics`)
  and M5 (`alertes_externes`, fed by `windfarm.farm_alerts` in `evaluer_modele` when a
  farm is declared; `/api/wind-farm` then shows the journal items). Lifecycle in a JSONL journal
  (`run_dir/logs/alertes.jsonl`, chronological instants only). **Opt-in**
  (`alerts.enabled: false`): never changes the served value; `/api/alerts` answers `409`
  when disabled. `an1.py` simulates drifts/outages; detector, κ and input rule are chosen on
  **validation** only from fixed grids (`charger_seuils` refuses test) by
  `scripts/m4_an1_colab.py`. AN1 v1 was measured NON-conforming on 2026-10-16 (test read
  once); v2 was designed after that, so its run (`--deuxieme-lecture`) is a second reading,
  not an independent measure. See `docs/ALERTES.md` (measurement history, never rewritten).
- **`reconstruct.py` / `export.py`** — npz + ANEO-shaped CSV + figures; `export_web` writes raw
  Float32 binaries plus `manifest.json` for `web/index.html`.
- **`windfarm.py`** (lot M5, `docs/EOLIEN.md`) — opt-in via `wind_farm.turbines` (declaration
  YAML, example in `configs/eolien/parc_demo.yaml`). Pure post-processing of a `Volume`: hub
  wind interpolated in height **above ground** (levels are ASL: h = z − terrain), REWS,
  shear α, veer, IEC density-normalised power, farm p10/p90 over the M2 ensemble members
  (captured at the turbine columns only, `Volume.farm_context`), alerts, Weibull/AEP/wind
  rose, and EOL1 (`scripts/m5_eol1_colab.py`). Exposed as `GET /api/wind-farm`, MCP
  `production_eolienne`, CLI `wind-farm` / `windrose`.
- **`publish.py` / `export_noeuds.py`** (lot M6, HIS1/DE3, `docs/PUBLICATION.md`) — a
  `Volume` (+ M2/M4/M5 when active, else the B5 nulls) becomes one ATOM rev. 2 snapshot
  laid out exactly like `contract/fixtures/<case>/` (validated by the contract's
  `validate.mjs`, copied in `tests/data/atom_rev2/tools/`). The B1 mask is recomputed on
  the float32 terrain (`NaN <=> z - terrain < min_agl_m`). `DepotLocal` writes atomically
  (temp dir + rename, `latest.json` last, never backwards); `ClientPublication` follows
  `publish.mjs` (model once, files, commit, heartbeat) — the publish token comes ONLY from
  `ATOM_PUBLISH_TOKEN`, never argv/logs, and redirects are never followed. A commit counts
  only once `/v1/times` and `/v1/latest` show it (the current Worker answers "unchanged"
  without repairing a crash after `_commit.json`): else `PublicationIncomplete`, nothing
  marked or purged. Publishing REFUSES a config without `real.origin_lat/lon`. HIS1 is
  tested against `tests/worker_simule.py` (Python copy of the Worker's publish routes,
  with connection cuts and the post-`_commit.json` failure). NetCDF/CSV exports (spec 8.4)
  are made from the PUBLISHED folder, checked by an output schema before delivery.
- **`baselines.py` / `preuves.py` / `sobriete.py`** (lot M8, `docs/PREUVES.md`) — reference
  methods (log profile / power law × IDW / ordinary kriging, numpy-scipy, no PyKrige),
  variogram and α fitted on **validation** only (`charger_ajustement` refuses test), and
  the network studies E1 (density), E2 (placement by M2 uncertainty), E3 (backup sensor),
  E5 (CodeCarbon → nvidia-smi → declared watts → duration only; never an invented
  number). Every method is scored by `evaluate.metriques_loader`, the metric code
  extracted from `Evaluator` (one pass over the loader, all predictors on the same
  instants/sensors/nodes). GPU run: `scripts/m8_preuves_colab.py` (new session folder);
  CLI `pcfno3d baselines`. Data/model cards: `docs/DATA_CARD.md`, `docs/MODEL_CARD.md`;
  UML (PlantUML) in `docs/uml/`.

## The service layer (`service/`, `webapp/`)

Thin by design: physics and numerics belong in `pcfno3d`, never here.

- **`app.py`** — `create_app()` is a **factory, not a module-level app**: the MCP session
  manager can only be run once per instance, so every startup (uvicorn, each test) builds its
  own. `app = create_app()` at the bottom is what uvicorn imports. Routes live on a module-level
  `APIRouter` with prefix `/api`; `/app/` serves `webapp/`; `/mcp` mounts the MCP sub-app.
- **`auth.py`** — API keys as `pcf_<32>`, stored **only as SHA-256** in
  `run_dir/api_keys.json` (gitignored). Scopes: `read`, and `admin` which covers `read`.
  **Closed by default** (lot M7): no keys at all = everything `401` except `/api/health`
  (`"auth": "aucune_cle"`). Open mode only by explicit opt-in (`serve --insecure-open` /
  `PCFNO3D_INSECURE_OPEN`); even then only direct loopback requests (127.0.0.1/::1, no
  `X-Forwarded-For`/`Forwarded`/`CF-Connecting-IP`/`X-Real-IP`) are served without a key
  (`app.requete_locale`), `serve` refuses a non-loopback `--host` whether the opt-in comes
  from the flag or the env var, and `colab.py` never passes it on (`PCFNO3D_FORBID_OPEN`). The store re-reads the file when its stat signature changes, so
  a CLI revoke takes effect on the next request; the server never writes `api_keys.json`
  while verifying (last use goes to `api_keys_usage.json`) so it can't clobber a revoke.
  One middleware (`app.make_guard`) does key check, per-key quota (`quotas.py`, in-memory,
  429 + `Retry-After`) and the JSONL access log (`journal.py`, never a secret) for `/api`
  and `/mcp`; route `Depends` only check scope. `/api/health` is public — the web app calls
  it before asking for a key.
- **`mcp_server.py`** — one `MCPServer` with 10 tools over the same `WindFieldModel`.
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

**Real data (`realdata.py`, `docs/REAL_DATA.md`):** adapters write real sources into the SAME
cache format (S, Y, index). A target may be *partial*: **NaN = not observed**; losses,
`dso.measure`, `evaluate`, `normalize` and `train.validate` only count known nodes, so the
synthetic data (no NaN) gives identical numbers. `Y[3]` (rho) must stay finite everywhere
(mass residual) — adapters fill it from ISA. Nothing real is imported yet: the domain has no
geographic anchor (`real.origin_lat/lon` is None and the code refuses to guess).

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
