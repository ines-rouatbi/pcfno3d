# Rapport de la nuit du 11/11 (accord écrit d'Ines donné à l'avance)

État à 03:40. Branche `agent/ines/os7-national` (dépôt public `ines-rouatbi/pcfno3d`), rien fusionné sur `main`.
Valeurs de **validation** = vérité d'entraînement v3 (générateur forcé par ERA5), jamais appelée « mesures » ; les
METAR et radiosondages sont les mesures. **Aucun test n'a été lu cette nuit.** Rien n'a été supprimé ni écrasé
sur le Drive ou dans `runs/` ; aucune règle hachée ni aucun seuil de DSO modifié.

## 0. En une phrase

Les contrôles des capteurs montrent qu'il n'y a pas de bug de branchement (Grand Tunis v4 : les capteurs arrivent au
réseau, qui y répond à ~1,5 %) ; les graines 42 de C' et D sont entraînées ; les graines 43 ont été coupées par Colab
vers 02:45 (A100 retirées) et la relance s'est arrêtée deux fois au montage du Drive (clic « Continuer » requis) :
**la suite GPU (graines 43-44, test v4, D2, moyenne des graines, contrôle national) attend un clic le matin.**

## 1. Contrôle du branchement des capteurs

Script `scripts/controle_capteurs.py` (`3f2e442`), carnets `notebooks/current/pcfno3d_controle_capteurs_{national,grand_tunis}_colab.ipynb`
(`8331b6d`), test unitaire `tests/test_forcage_era5.py::test_capteurs_arrivent_au_reseau` (une entrée modifiée aux
capteurs change la sortie près des capteurs, par le chemin de l'évaluation et par celui de l'OS6 ; sortie identique = échec).

### Grand Tunis v4 (modèle livré, chemin de l'OS6) — 3 instants de validation (Colab CPU), m/s

| Instant | Masque avant → après 50 capteurs | Ajout des 50 capteurs (valeur vraie) : max \|Δu\| < 5 km | +15 m/s sur u aux 50 capteurs : max \|Δu\| < 5 km / ailleurs ; max \|Δv\| < 5 km | +15 m/s aux 3 stations : max \|Δu\| < 5 km | Même test, chemin OS6 | Masque seul ajouté (OS6 Grand Tunis) |
|---|---|---|---|---|---|---|
| 0001_000000 | 597 → 647 | 0,134 | 0,218 / 0,007 ; 0,179 | 0,032 | 0,218 | 0,108 |
| 0001_060000 | 597 → 647 | 0,195 | 0,259 / 0,007 ; 0,151 | 0,030 | 0,259 | 0,108 |
| 0001_180000 | 597 → 647 | 0,194 | 0,199 / 0,009 ; 0,127 | 0,031 | 0,199 | 0,163 |

Chemin : les 50 capteurs sont dans le masque (+50 cellules), leur valeur est écrite exactement dans S (écart 0,0), le
tenseur d'entrée du réseau a le masque à 1 et la bonne valeur normalisée à ces cellules (écart 0,0). Les chemins de
l'évaluation et de l'OS6 donnent exactement les mêmes chiffres. Sortie : `MyDrive/pcfno3d_grand_tunis/runs/grand_tunis_v4_dso/controle_capteurs.json`.

**Conclusion (Grand Tunis) : pas de bug de branchement.** La sortie répond localement (< 5 km) mais très faiblement :
+15 m/s en entrée → 0,20 à 0,26 m/s en sortie (≈ 1,5 %), < 0,01 m/s ailleurs. Le modèle se fie à l'ERA5 dense et
traite un point de mesure comme une petite correction locale : c'est ce qui rend la courbe de l'OS6 du Grand Tunis
plate au 4e chiffre (0,5326 m/s), pas un défaut de code. **Rien à refaire pour un bug** (ni D, ni l'OS6 du Grand
Tunis) ; il faut un modèle qui exploite les capteurs (D2, ci-dessous).

### National (C' et D graine 42, + lots d'entraînement de D) — **pas encore lancé**

Le carnet est prêt ; il demande une session GPU (voir § 5). Sur le mini-jeu, le même contrôle passe (capteurs dans le
masque et dans le tenseur d'entrée, sortie modifiée près d'eux, ~1e-4 m/s pour +15 m/s sur un petit modèle).
Conformément à ta consigne, **D graine 44 n'a pas été lancé** tant que ce contrôle n'est pas fait.

## 2. Entraînements v4 (règle `60671e84…`) — validation seulement

| Entraînement | État | Meilleure époque | Score (val_score) | RMSE max (niveau ASL) |
|---|---|---|---|---|
| C' s42 | fini (30 ép.) + éval. d'information | ≥ 18 (à lire dans `history.json`) | 0,614 à l'ép. 18 | 0,307 m/s (2500 m) à l'ép. 18 |
| D s42 | fini (30 ép.) + éval. d'information | 27 | 2,317 | 1,159 m/s (25 m) |
| C' s43 | **coupé à l'ép. 3** (session Colab perdue vers 02:45) | — | 1,367 à l'ép. 3 | 0,683 m/s (650 m) |
| D s43 | **coupé au début** (même moment) | — | — | — |
| C' s44, D s44 | non lancés | | | |

Départ commun (v2 livré sur la validation v3) : score 3,695, RMSE max 1,848 m/s à 650 m. D s42 : plateau en fin
(2,317 → 2,330 aux ép. 28-30, l'ép. 27 est gardée).

Évaluation d'information (sans effet sur le choix ; `rmse_max_uv` = pire tranche 0-1 km au-dessus du sol, 64 instants) :

| Entrée | C' s42 | D s42 | ERA5 seule |
|---|---|---|---|
| ERA5 propre (production) | **0,253** | 0,940 | 0,323 |
| ERA5 dégradée | 1,898 | **1,354** | 1,893 |
| ERA5 dégradée + 50 capteurs simulés | 1,898 | 1,355 | 1,893 |

Lecture : C' fait mieux que l'ERA5 seule sur l'entrée de production (validation) ; D corrige l'ERA5 dégradée
(−28 %) mais perd sur l'ERA5 propre ; les 50 capteurs ne changent presque rien (cohérent avec le § 1). Détail :
`docs/resultats/national/v4_entrainements_validation.md`.

**Test v4 : non lu** (il faut les 6 entraînements). **Verdict de la règle v4 : aucun** pour l'instant.

Les dossiers interrompus `runs/national_1km_v4_{Cp,D}_s43` sont sur le Drive tels quels ; les carnets v4 régénérés
(`eb3b718`, code toujours figé au commit `77a058b` des graines 42) les **mettent de côté** à la relance (renommés
`…_interrompu_<date>`, jamais supprimés ni écrasés) puis réentraînent sous le nom de la règle.

## 3. Ce qui est codé et testé (sans calcul sur données réelles sauf mention)

| Travail | Règle (sha256) | Commit | État |
|---|---|---|---|
| D2 (canal d'innovation, ta définition ; référence « interpolation optimale seule ») | `regle_d2_national.json` `c7c98a20…` (choix par défaut de l'agent, listés et justifiés dans la règle) | `5767232` (+ carnets `6e732f0`), fusionné `d756fb1` | code + 10 tests mini-jeu passés ; 3 carnets d'entraînement + carnet de test prêts ; **non lancé** (GPU) |
| Moyenne des 3 graines de C' (validation, DSO4, dispersion DSO6) | `regle_ensemble_cp.json` `271ddc61…` | `bb991bf` (+ `14b2198`) | code + 5 tests ; **non lancé** (attend C' s43/s44) |
| Stations 2015-2022 (ISD + SYNOP/OGIMET) | — | `593d887`, fusionné `8be776d` | **fait** : 29 stations tunisiennes, 13 enregistrements des 12 aéroports, **16 stations de validation supplémentaires** retenues (≥ 50 % de créneaux valides en 2022 après contrôle) ; contrôle qualité (pannes, valeurs figées, calmes suspects, abris signalés) ; descripteurs de site (z0, pente, TPI 5/20 km, distance à la côte) ; ISD 647 Mo hors git. `docs/resultats/national/stations_supplementaires.md` |
| MOS de l'erreur ERA5 10 m | `regle_mos_era5.json` `80fe361d…` (fixée avant tout calcul) | `f3f826f`, `32f8206` | code + tests synthétiques ; **non calculé** : le téléchargement ERA5 (Open-Meteo, 71/280 séries) a été arrêté par Claude Code faute de mémoire sur ines-pc ; non relancé (consigne : ne pas relancer sans demande) |
| Phases 4-8 (GFS national/OS7, OS6 national sur D, agriculture et éolien nationaux, ATOM Wind national + onglet « Réseau ANEO ») | OS7 `4360860…`, OS6 `c705aa8b…`, agri `739467e…`, éolien `ff0b869…` (PROPOSITIONS) | `cff9457`, `029620c`, `c3fda79` | code + tests mini-jeu ; aucun téléchargement GFS (pas de session) |

Suite complète (`python -m pytest -q`) : passe sur `8be776d`.

## 4. Problèmes rencontrés

1. **Colab** : 2 sessions GPU au plus ; les deux sessions des graines 43 se sont arrêtées vers 02:45 (C' à l'ép. 3,
   D au départ). À la relance, « le type de GPU sélectionné n'est pas disponible » : **plus d'A100**, Colab donne un
   L4 (24 Go ; le national a besoin de ~29 Go sur A100 sans recalcul d'activations). Cause probable : retrait des A100
   (quota / disponibilité) ; l'ouverture d'une 3e session (CPU, contrôle du Grand Tunis) à ce moment n'est pas exclue.
2. **Montage du Drive** : à chaque nouvelle machine, Google demande « Continuer » dans une fenêtre que je ne peux pas
   cliquer ; deux échecs de suite sur C' s43 → étape arrêtée (règle des deux échecs), session L4 libérée.
3. **Mémoire d'ines-pc** : Claude Code a arrêté des tâches de fond (téléchargement ERA5 du MOS, une minuterie) ;
   rien de perdu, le téléchargement reprend où il s'est arrêté.
4. **Affichage Colab** : les pages se déconnectent (« Reprise de l'exécution… ») alors que l'entraînement continue ;
   suivi par les fichiers du Drive (`history.json`, journaux).
5. Corrigé en route : le carnet de contrôle v3 échouait sur `pip install -e` (dossier courant effacé) — `cc8aea6`.

## 5. Ce qui reste à faire / à décider le matin

1. **Un clic** : lancer les carnets et cliquer « Continuer » au montage du Drive. Ordre proposé (2 sessions) :
   contrôle national des capteurs + C' s43 ; puis D s43, C' s44, D s44 (si le contrôle national ne trouve pas de bug) ;
   puis test v4 (lecture unique, `pcfno3d_national_v4_test_colab.ipynb`) ; puis D2 s42 (écrit L, r, H) puis s43/s44 ;
   moyenne des graines de C' (validation). Sur L4, compter ~5 h par entraînement au lieu de ~2 h 15.
2. **A100 ou L4** : accepter L4 (plus lent, recalcul d'activations) ou attendre le retour des A100.
3. **Relancer le téléchargement ERA5 du MOS** (`python -I scripts/mos_era5.py era5 --stations artifacts/national/mos/stations_era5.csv --sortie artifacts/national/mos/era5_openmeteo`, ~80 min), puis préparer → choisir (2021) → test 2022 seulement si la règle est remplie.
4. Choix par défaut de D2 à confirmer (liste dans `regle_d2_national.json`, champ `choix_par_defaut`), notamment :
   capteurs simulés à 10 m, OI gaussienne coupée à 4 L, L borné [20 ; 50] km (risque de borne atteinte avec des
   aéroports distants de dizaines de km), H ∈ {200, 500, 1000} m choisi sur la validation avec l'OI seule.
5. Règles en PROPOSITION des phases 4-8 (OS7, OS6, agri, éolien) : à valider ; maquette `reseau_aneo.html` jamais
   reçue (onglet provisoire construit d'après ta description).
6. Les 16 stations supplémentaires en entrée du modèle : décision à toi (aujourd'hui : validation seulement).
