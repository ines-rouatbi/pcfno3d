# Model card — PC-FNO3D v7

*Livrable L7 (cahier des charges v9, § 9.1 p. 21 ; tâche 13 p. 44). Chaque chiffre est
suivi de sa source (dans le dépôt, ou dossier de session sur le Drive
`aneo_fno3d_runs/v7/…`). Les mesures GPU des 15-16/10/2026 (Colab L4) sont reportées ; un
critère encore sans mesure est marqué « en attente » : aucun n'est présenté comme atteint.
Synthèse : `docs/BILAN_FINAL.md`.*

## Identité

| Élément | Valeur | Source |
|---|---|---|
| Modèle servi | `best_dso_v7.pt` = modèle du BLOC 9 du notebook v7, **époque 140**, score de sélection 1,19523 | `tukhnanutha.md` § 5 (« v7 measured results », « v7 served locally ») |
| Notebook de référence | `notebooks/current/fno3d_v7.ipynb` (avec ses sorties Colab) ; code identique bloc à bloc à `fno3d_v7_1.ipynb`, porté dans le paquet `pcfno3d` 0.4.0 | `CLAUDE.md` ; `src/pcfno3d/__init__.py` |
| Fidélité du portage | le paquet retrouve la table DSO1 du notebook à **0,0000 m/s** près aux 12 niveaux (Colab, 2026-10-12) ; 453,3 M paramètres chargés en `strict=True` | `docs/REPRODUCTIBILITE.md` ; `tukhnanutha.md` § 5 |
| Taille du checkpoint | 1,81 Go | `CLAUDE.md` |
| Empreinte sha256 des poids | `b2529442ee6fcdb18d3f35fe176663022604395743c161a25541ccd4e365e7de` (sha256 des octets de `best_dso_v7.pt`, 1 813 028 515 octets ; D1) | `versions.model_sha256` du `latest.json` et des `models/b2529442/meta.json` publiés par M6 dans le dépôt local `AppData/Local/pcfno3d/runs/v7/publication/` (8 instantanés du 2025-02-28, générés le 2026-10-15), calculé par `publish.versions_modele` sur le fichier servi ; préfixe identique à `docs/REPRODUCTIBILITE.md` |
| `norm_stats.json` | fait partie du modèle (dénormalisation) ; sha256 `7bae6748f270fa5e3f08a69a7f8e919f01d641d29b125981168cc15a03ee25f7` (1 038 octets) | `versions.norm` publié = `7bae6748` (sha8, même dépôt local) ; empreinte complète calculée le 2026-10-16 sur `AppData/Local/pcfno3d/runs/v7/norm_stats.json`, préfixe identique au sha8 publié |
| Statut | **v7 figée** après M1 (2026-10-16, tag `v7-fige`), seule, sans correction post-modèle ; elle reste le modèle servi (la v8, affinage, n'est pas meilleure) | `docs/resultats/modele_fige/README.md` ; `docs/M1_PROTOCOLE.md` ; `docs/rapport_dso1.md` § 3 |

## Architecture

Opérateur neuronal de Fourier 3D contraint par la physique : convolutions spectrales
`SpectralConv3d` sur (z, y, x), branche convolutive locale, tête de projection ; deux îlots
fp32 volontaires (FFT et tête), le reste en bf16 sur GPU. Source : `CLAUDE.md`
(`model.py`) ; cahier p. 25 (M2).

| Hyperparamètre | Valeur | Source |
|---|---|---|
| Largeur | 48 | `configs/local_v7.yaml` |
| Modes (z, y, x) | 6, 32, 32 | idem |
| Couches spectrales | 4 | idem |
| Rembourrage | 4, 8, 8 | idem |
| Couche cachée de la tête | 128 | idem |
| Branche locale, regroupement 2 × 2 des capteurs | oui, oui | idem |
| Paramètres | 453,25 M | `configs/local_v7.yaml` (commentaire) |
| Entrée | 25 canaux : 7 mesures de surface normalisées + disponibilité + 6 canaux de relief + 3 de géométrie par niveau + 8 temps / physique (heure, jour, vent géostrophique et thermique ajustés sur les capteurs visibles) | `src/pcfno3d/geometry.py` (`c_in`), `features.py`, `dataset.py` |
| Sortie | u, v, w sur 12 niveaux × 250 × 500 (≈ 1,5 M nœuds par instant) | `tukhnanutha.md` § 5 |

## Entraînement

| Élément | Valeur | Source |
|---|---|---|
| Données | `aneo_dataset_3d_v2`, jours d'entraînement (voir `docs/DATA_CARD.md`) | — |
| Époques × pas | 140 × 400, lot 1, accumulation 2 | `configs/local_v7.yaml` |
| Optimisation | AdamW, lr 1·10⁻³ (OneCycle), décroissance 1·10⁻⁴, écrêtage 1,0, moyenne exponentielle des poids 0,999 (poids EMA évalués et servis) | `configs/local_v7.yaml` ; `src/pcfno3d/train.py` (portage du BLOC 9) |
| Perte | L_data (w pondéré par niveau, plafond 25) + 0,15·L_grad + φ(t)·[1,0·L_mass / 5·10⁻⁷ + 0,1·L_bc], montée de la physique sur 10 époques | `configs/local_v7.yaml` ; `src/pcfno3d/losses.py` |
| Masquage | fraction de capteurs visibles tirée dans [0,2 ; 1,0] à chaque exemple | `configs/local_v7.yaml` (`mask.keep_train`) |
| Normalisation | statistiques des jours d'entraînement seulement | `CLAUDE.md` |
| Matériel | NVIDIA A100-SXM4-80GB (Colab) | sortie de `fno3d_v7.ipynb` (BLOC 0) |
| Durée journalisée | époques 61 à 140 : 9 582 s (119,8 s par époque) ; époques 1 à 60 non journalisées | sortie du BLOC 9 de `fno3d_v7.ipynb`, lue par `sobriete.historique_entrainement` (`docs/PREUVES.md`, E5) |
| Énergie, CO₂ | entraînement **non mesuré** (puissance non journalisée) ; inférence : 9,3 µWh GPU par reconstruction (E5, L4, nvidia-smi, GPU seul) ; CO₂ non calculé (intensité carbone non fournie) | `docs/PREUVES.md` (E5) ; `aneo_fno3d_runs/v7/preuves_pcfno3d/best_dso_v7/session_20261016T125440Z` |
| Diagnostic | pas de surapprentissage : pertes de données entraînement / validation 0,0050 / 0,0053 en fin d'entraînement | `tukhnanutha.md` § 5 |

## Résultats (test synoptique : 120 instants, 50 % des capteurs visibles, une passe)

Source : `docs/rapport_dso1.md` § 2 (table du BLOC 10, retrouvée par le paquet), sauf mention.

| Critère | Seuil (cahier p. 21–22) | Mesure v7 | État |
|---|---|---|---|
| DSO1 RMSE u, v par niveau | ≤ 0,5 m/s | max u **0,665** m/s (1 000 m) ; u 0,564 à 650 m, 0,632 à 800 m ; v 0,508 à 1 000 m | **non conforme** (650–1 000 m) ; la correction M1 n'est pas retenue (ci-dessous) |
| DSO1 RMSE global | ≤ 0,5 m/s | u 0,542, v 0,437 | **non conforme** (u) |
| DSO1 R² u, v par niveau | ≥ 0,95 | min 0,9911 | conforme |
| DSO2 résidu de continuité | ≤ 1·10⁻⁴ | 2,58·10⁻⁵ | conforme |
| DSO3 rugosité | dans [0,9 ; 1,1] | 0,951 | conforme |
| DSO4 inférence (GPU) | < 1 s | 91 ms (A100, notebook v7, `tukhnanutha.md` § 5) ; 104 ms (campagne v8, `docs/rapport_dso1.md`) ; L4, valeur servie seule (fiabilité différée) : **0,440 s**, 0,444 s avec la fiabilité active ; délai de la fiabilité 3,834 s (lot 4), sans verdict (`aneo_fno3d_runs/v7/fiabilite_pcfno3d/best_dso_v7/session_20261016T134302Z/resume_dso4.json`) | conforme (valeur servie) ; la fiabilité synchrone (4,2–4,7 s) était non conforme (`docs/FIABILITE.md`) |
| DSO5 RMSE avec 25 % des capteurs | ≤ 0,7 m/s | 0,646 m/s | conforme |
| DSO6 fiabilité des indicateurs | couverture 80 % et corrélation > 0 | 1re lecture (15/10, q 0,8) : couverture u 0,751, v 0,811, w 0,794, vitesse 0,796 ; Spearman 0,289 / 0,307 / 0,229 / 0,290 (`aneo_fno3d_runs/v7/fiabilite_pcfno3d/best_dso_v7/session_20261015T234810Z`). 2e lecture (16/10, q = 0,90 choisi sur la validation) : couverture 0,871 / 0,912 / 0,896 / 0,904 ; Spearman 0,295 / 0,311 / 0,231 / 0,291 (`aneo_fno3d_runs/v7/fiabilite_pcfno3d/best_dso_v7/session_20261016T115325Z`) | **non conforme** (1re lecture) ; **conforme en seconde lecture**, non indépendante du test (`docs/FIABILITE.md`) |
| AN1 alertes | ≥ 90 % de détection, < 5 % de fausses alertes | v1 (16/10/2026, test) : dérives détectées 5,2 %, 0 % de fausses alertes ; entrées : fausses alertes 6,7 % | **non conforme** (v1, `aneo_fno3d_runs/v7/alertes_pcfno3d/best_dso_v7/session_20261016T121344Z`) ; v2 (16/10, test) : dérives détectées 100 %, fausses alertes 2,2 %, entrées : détection 100 %, fausses alertes 0 % → **conforme en seconde lecture**, non indépendante du test (`aneo_fno3d_runs/v7/alertes_pcfno3d/best_dso_v7/session_v2_20261016T142959Z`, `docs/ALERTES.md`) |
| EOL1 production éolienne | écart < 5 % | parc **de démonstration** (3 éoliennes, 12 MW), 120 instants de test : écart relatif 1,71 % ; nMAE puissance 1,17 % (`aneo_fno3d_runs/v7/eolien_pcfno3d/…`) | **conforme sur le parc d'exemple** ; verdict officiel en attente du parc réel (`docs/EOLIEN.md`) |
| HIS1 historique | toute reconstruction passée retrouvée avec ses indicateurs et versions | publication et historique implémentés (lot M6 : `src/pcfno3d/publish.py`, R2, `latest`, `/v1/times`) ; atomicité testée sur CPU contre un Worker simulé | rejeu des 120 instants de test fait (16/10, L4, 9,1 s par instant, historique 120) ; 8 instants republiés dans le déploiement local (`docs/PUBLICATION.md`) |
| DE2 reproductibilité | même entrée et même version → même sortie | égalité bit à bit sur CPU (mini-modèle), test automatique | conforme sur le mini-modèle ; à rejouer sur v7 CPU (`docs/REPRODUCTIBILITE.md`) |

**Cinq objectifs DSO sur six mesurés atteints (DSO2 à DSO5 et le volet R² du DSO1) ; DSO1
échoue en RMSE entre 650 et 1 000 m** — de 0,06 à 0,17 m/s au-dessus du seuil
(`docs/rapport_dso1.md` § 2). Le critère complémentaire **proposé** (tolérance OMM,
max(0,5 m/s ; 10 % de la vitesse moyenne du niveau)) est respecté à tous les niveaux ; il
n'est pas validé par Tukhnanutha et ne remplace pas le verdict officiel
(`docs/rapport_dso1.md` § 5).

Tentatives de correction du DSO1, toutes sans succès (`docs/rapport_dso1.md` § 3) :
moyenne de 8 passes (pire à tous les niveaux : 1 000 m u 0,672 ; DSO4 930 ms), affinage v8
(1 000 m u 0,649 / v 0,529 ; RMSE vectorielle à 1 000 m 0,837 m/s comme la v7). Hypothèse
la plus compatible : limite d'information (un seul plan géostrophique relie les capteurs à
25–30 m à l'atmosphère libre) ; non démontrée.

**M1, dernière tentative bornée (mesure décisive 10i, 2026-10-16, A100-SXM4-80GB)** :
correction post-modèle du vent moyen par niveau (estimateur géostrophique v2 + ridge par
niveau, alpha 0,01), jugée par une règle pré-enregistrée, C1 ET C2 ET C3
(`docs/M1_PROTOCOLE.md`, section « Résultat de la mesure décisive ») :

- C1, test synoptique : pire niveau u 0,665 → 0,419, v 0,508 → 0,382 ; R² min 0,9933 →
  **conforme** ;
- C2, 46 instants d'épisodes jamais utilisés : la correction dégrade v à 150, 200 et 1000 m
  et u à 650 m au-delà des seuils de la règle (pire niveau u 1,148 → 1,056, v 1,068 →
  1,104) → **non conforme** ;
- C3 : DSO2 2,58e-5, DSO3 0,951, DSO5 0,508, DSO4 121 ms (modèle seul 92 ms) → **conforme**.

**Correction non retenue : la v7 est gelée seule** (`docs/resultats/modele_fige/README.md`). Le
DSO1 reste non atteint à 650–1 000 m et se rapporte avec le critère OMM. Réserve imprimée par
10i : les coefficients sont appris sur synthétique ; sur réel, l'erreur des baromètres
(±10 à 50 Pa) est du même ordre que le signal géostrophique (26 Pa pour 10 m/s).

Robustesse (BLOC 10) : RMSE max u 0,828 / 0,646 / 0,593 / 0,705 m/s à 10 / 25 /
50 / 100 % des capteurs — anomalie : pire avec tous les capteurs qu'avec la moitié
(`tukhnanutha.md` § 5 ; sortie du BLOC 10 de `fno3d_v7.ipynb`).

**Comparaison aux méthodes de référence (L6) et études du réseau (L9)** : mesurées le
16/10/2026 sur L4 (`aneo_fno3d_runs/v7/preuves_pcfno3d/best_dso_v7/session_20261016T125440Z`, détail dans `docs/PREUVES.md`). Sur le test (120
instants, 50 % des capteurs), RMSE max u, v : PC-FNO3D 0,665 m/s contre 5,027 à 6,487 m/s
pour les quatre références (profil log ou loi de puissance × IDW ou krigeage). E1 :
PC-FNO3D tient DSO5 jusqu'à 50 % des capteurs. E2 : **résultat négatif** (le placement guidé
par σ fait pire que le hasard). E3 : sur ce treillis dense, krigeage et IDW remplacent mieux
un capteur en panne que PC-FNO3D.

## Limites

- **Entraîné et évalué uniquement sur des données synthétiques** ; transférabilité au réel
  non démontrée (cahier p. 41, § 18 ; `docs/REAL_DATA.md`).
- **DSO1 non atteint** au-dessus de 500 m (voir ci-dessus) ; la dernière tentative (M1) n'a
  pas été retenue et le modèle est figé.
- **Densité réaliste** : premier test indicatif (8 instants d'un jour, CPU) — à un capteur
  par km, l'erreur des niveaux bas est multipliée par 2 à 3 ; u à 1 000 m ≈ 0,87–0,91 m/s
  quelle que soit la densité (`tukhnanutha.md` § 5). À confirmer sur 120 instants (E1).
- **Tous les capteurs** (cas de production) : erreur plus forte qu'à 50 % (robustesse
  ci-dessus) ; le modèle a été entraîné avec une fraction tirée dans [0,2 ; 1,0].
- **CPU** : 15 à 25 s par instant (13,7 s mesurées sur le poste de développement) contre
  91 ms sur GPU ; DSO4 n'est tenu que sur GPU (`CLAUDE.md` ; `configs/local_v7.yaml`).
- **Non-déterminisme GPU** (bf16, cuDNN) : quelques 10⁻³ m/s entre deux exécutions GPU
  (`docs/REPRODUCTIBILITE.md`).
- Le vent sous le relief n'existe pas (NaN) ; niveaux 25 à 75 m exclus des verdicts
  (couverture < 5 %).

## Usages

- **Prévus** : démonstrateur de reconstruction 3D du vent dans la couche limite sur le
  domaine pilote synthétique ; évaluation des critères du cahier ; support de la chaîne
  fiabilité / alertes / éolien ; préparation d'un pilote de terrain (cahier p. 22, § 9.4).
- **Déconseillés** : décision opérationnelle ou de sécurité (arrêt d'éoliennes, travaux en
  hauteur, grues) sur la seule foi du modèle ; dimensionnement ou investissement éolien sans
  campagne de mesure (cahier p. 35, § 16.5) ; usage sur un domaine, une grille ou des
  données réelles différents de ceux de l'entraînement sans réentraînement et validation ;
  lecture des valeurs à 650–1 000 m comme conformes au DSO1.

## Reproduire

```bash
python -m pcfno3d -c colab evaluate          # table DSO1-DSO5 (GPU)
python scripts/m2_calibration_colab.py -c colab   # DSO6
python scripts/m4_an1_colab.py -c colab --deuxieme-lecture   # AN1 (v2)
python scripts/m5_eol1_colab.py -c colab          # EOL1
python scripts/m8_preuves_colab.py -c colab       # L6, E1, E2, E3, E5
```
