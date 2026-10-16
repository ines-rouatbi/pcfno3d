# Modèle figé — PC-FNO3D v7 (gel après M1, 2026-10-16)

*Gel prévu au § 8 de `docs/M1_PROTOCOLE.md`, cas « correction rejetée ». Après ce gel, le
modèle ne change plus dans le cadre du PFE. Tag git annoté : **`v7-fige`** (nom retenu par
l'utilisatrice ; le § 8 prévoyait `modele-fige-v7`).*

## Modèle gelé : v7 seule

| Élément | Valeur |
|---|---|
| Poids | `best_dso_v7.pt`, modèle du BLOC 9 du notebook v7, époque 140, score de sélection (validation) 1,19523 |
| sha256 des poids | `b2529442ee6fcdb18d3f35fe176663022604395743c161a25541ccd4e365e7de` |
| Normalisation | `norm_stats.json`, sha256 `7bae6748f270fa5e3f08a69a7f8e919f01d641d29b125981168cc15a03ee25f7` |
| Correction post-modèle | **aucune** (non retenue par M1) ; elle reste sur la branche `agent/ines/m1-correction-post-modele`, non fusionnée |
| Code qui sert et évalue | paquet `src/pcfno3d/` au commit du tag ; configuration `configs/local_v7.yaml` (local), `configs/colab.yaml` (Colab) |
| Notebook de référence | `notebooks/current/fno3d_v7.ipynb` (avec ses sorties Colab) |
| Notebook de la mesure M1 | `notebooks/current/m1_bloc10i_colab.ipynb` (code identique à celui exécuté, sorties vidées ; `python scripts/make_m1_colab_nb.py --check` le vérifie, sha256 du code 10i `a1fcc878ed3c6c6047538e5b48df25c6633c972a413d6bd42f415926599fbd6f`) |

`SHA256SUMS` (ce dossier) liste les empreintes, avec des chemins relatifs au dossier du run v7.
Vérification : `cd <dossier v7> && sha256sum -c <dépôt>/docs/resultats/modele_fige/SHA256SUMS`.
Sur la copie locale (`AppData/Local/pcfno3d/runs/v7/`), les trois lignes sont `OK` (2026-10-16).

## Tableau DSO final (test synoptique, 120 instants, 50 % des capteurs)

Source : `docs/BILAN_FINAL.md` § 1.

| Critère | Seuil | v7 | Verdict |
|---|---|---|---|
| DSO1 RMSE u, v par niveau | ≤ 0,5 m/s | max u 0,665 (1 000 m), v 0,508 (1 000 m) | **non conforme** (650–1 000 m) |
| DSO1 RMSE global | ≤ 0,5 m/s | u 0,542 ; v 0,437 | **non conforme** (u) |
| DSO1 R² u, v (et w) | ≥ 0,95 | min 0,991 (w 0,966) | conforme |
| DSO1 révisé proposé (tolérance OMM) | max(0,5 ; 10 % de la vitesse) | — | conforme (`docs/rapport_dso1.md`) |
| DSO2 résidu de masse | ≤ 1e-4 | 2,58e-5 | conforme |
| DSO3 rugosité | [0,9 ; 1,1] | 0,951 | conforme (relu dans [0,9 ; 1,1]) |
| DSO4 inférence (valeur servie) | < 1 s | 0,440 s (L4), 0,444 s avec la fiabilité active ; 91 ms (A100, notebook) ; 92 ms (A100, modèle seul, session M1) | conforme |
| DSO5 RMSE à 25 % des capteurs | ≤ 0,7 m/s | 0,646 | conforme |

## Décision M1 (mesure décisive 10i, 2026-10-16, A100)

Détail : `docs/M1_PROTOCOLE.md`, section « Résultat de la mesure décisive (2026-10-16) ».

- **C1** (test synoptique) : **conforme**. Pire niveau après correction u 0,419, v 0,382 ;
  R² min 0,9933.
- **C2** (46 instants d'épisodes) : **non conforme** (v à 150, 200 et 1000 m, u à 650 m).
- **C3** (DSO2 2,58e-5, DSO3 0,951, DSO5 0,508, DSO4 121 ms) : **conforme**.
- **Décision : correction NON RETENUE ; la v7 est gelée seule.** Le DSO1 reste non atteint à
  650–1000 m et est rapporté honnêtement, avec le critère OMM (DSO1 révisé proposé, conforme).
- Réserve imprimée par 10i : coefficients appris sur synthétique ; sur réel, l'erreur des
  baromètres (±10 à 50 Pa) est du même ordre que le signal géostrophique (26 Pa pour 10 m/s).

## Emplacement des fichiers

Les poids (1,81 Go) ne vont pas dans git.

| Fichier | Drive (`MyDrive/aneo_fno3d_runs/`) | Local (ines-pc) |
|---|---|---|
| `best_dso_v7.pt` | `v7/best_dso_v7.pt` | `AppData/Local/pcfno3d/runs/v7/best_dso_v7.pt` (sha256 vérifié le 2026-10-16) |
| `norm_stats.json` | `v7/norm_stats.json` | `AppData/Local/pcfno3d/runs/v7/norm_stats.json` (sha256 vérifié le 2026-10-16) |
| Résultats du notebook v7 (BLOC 10) | `v7/eval_best_dso_v7/` (voir l'anomalie ci-dessous) | `AppData/Local/pcfno3d/runs/v7/eval_best_dso_v7/` |
| Résultats de M1 (10i) | `v7/eval_m1_bloc10i/decision_10i.json`, `decision_10i_par_niveau.csv` | copie ici : `eval_m1_bloc10i/` (sha256 vérifiés contre le `sha256.txt` de la session) |
| JSON de la correction (non retenue) | `v7/couche_limite_v7.json` (sha256 `c50928ae2a93dfb3363f5e8c307726bfa7c8321ab03ee30b6fa6aeef31487b10`), `v7/correction_moyenne_v7.json`, `v7/geostrophique_v2_v7.json` | — |

## Écart de sha256 de `test_summary.json` (contrôle M1-a) — expliqué, sans effet

- **Sur le Drive :** `v7/eval_best_dso_v7/test_summary.json` a pour sha256
  `28a5bbb8fb87b2c61f3c948760b899bb9c81590dd725c13b0550e6475c123894` (8 254 octets, 13/10 11:47).
- **Valeur publiée :** `7944ba0aa0311a08cd1e1555c1a225239b3deabc5c9c2b55dcf4406b5da17425`, celle de la
  copie **locale** (5 429 octets, 12/10).
- **Explication**, après comparaison des deux fichiers (16/10) :
  - La copie locale est la version **importée par le paquet** (`scripts/import_notebook_eval.py`) :
    valeurs arrondies au millième, plus des clés ajoutées (`produit_par`, `source`, `epoque`,
    verdicts `dso*_conforme`, `critere_omm`).
  - La copie du Drive est la sortie **brute** du BLOC 10 du notebook, ré-exécuté le 13/10 pendant
    les essais 10e à 10h, à pleine précision.
  - Les deux donnent les **mêmes métriques** aux arrondis près (u global 0,542 contre 0,5421558 ;
    DSO3 0,951 contre 0,9507193). Seul le temps DSO4 diffère (0,091 s contre 0,099 s).
  - La session M1 du 16/10 (A100) reproduit le fichier du Drive à l'identique, temps DSO4 excepté
    (`eval_m1_bloc10i/bloc10_rejoue/test_summary.json`).
- **Conclusion :** aucune valeur n'a été altérée, l'écart vient du format. `SHA256SUMS` retient la
  copie locale publiée et ajoute la copie brute du Drive.

## Ce que le § 8 demande encore (non faisable sans Drive ni GPU)

- [x] Copier ici `decision_10i.json` depuis `v7/eval_m1_bloc10i/` et l'ajouter à `SHA256SUMS`
      (fait le 16/10 : dossier `eval_m1_bloc10i/`, avec le CSV par niveau, la règle, le
      `sha256.txt` de la session et le rejeu du BLOC 10).
- [x] Ajouter à `SHA256SUMS` le sha256 complet de `couche_limite_v7.json` (lu dans le
      `sha256.txt` de la session).
- [x] Archiver le notebook **exécuté** (avec ses sorties), tel que la session l'a laissé :
      `m1_bloc10i_execute.ipynb` (copie du Drive du 16/10, 17 cellules avec sorties, sans secret).
- [ ] Constituer sur le Drive `aneo_fno3d_runs/fige/modele_fige_2026-10.zip` : poids,
      `norm_stats.json`, configuration, `decision_10i.json` et `SHA256SUMS`.
- [x] Investiguer l'écart de sha256 de `test_summary.json` sur le Drive : expliqué ci-dessus.
