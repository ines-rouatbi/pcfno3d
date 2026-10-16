# Bilan final — PC-FNO3D / ATOM (PFE)

*État au 2026-10-16. Chaque chiffre vient d'une mesure archivée (fichier cité) ; « en attente »
signifie qu'aucune mesure n'existe encore, jamais une valeur supposée.*

## 1. Le modèle v7 (figé après M1, tag `v7-fige`)

- **Poids :** `best_dso_v7.pt`, sha256
  `b2529442ee6fcdb18d3f35fe176663022604395743c161a25541ccd4e365e7de`.
- **Normalisation :** `norm_stats.json`, sha256
  `7bae6748f270fa5e3f08a69a7f8e919f01d641d29b125981168cc15a03ee25f7`.
- **Évaluation :** 120 instants synoptiques de test, 50 % des capteurs visibles. La chaîne du paquet
  reproduit au 1/1000 le tableau du notebook v7.

| Critère | Seuil | v7 | Verdict | Source |
|---|---|---|---|---|
| DSO1 RMSE u, v par niveau | ≤ 0,5 m/s | max u 0,665 (1 000 m), v 0,508 (1 000 m) | **non conforme** (650–1 000 m) | notebook v7 BLOC 10 ; M8 `preuves_pcfno3d/.../session_20261016T125440Z` |
| DSO1 RMSE global | ≤ 0,5 m/s | u 0,542 ; v 0,437 | **non conforme** (u) | idem |
| DSO1 R² u, v (et w) | ≥ 0,95 | min 0,991 (w 0,966) | conforme | idem |
| DSO1 révisé proposé (tolérance OMM) | max(0,5 ; 10 % de la vitesse) | — | conforme | `docs/rapport_dso1.md` |
| DSO2 résidu de masse | ≤ 1e-4 | 2,58e-5 | conforme | notebook v7 |
| DSO3 rugosité | [0,9 ; 1,1] | 0,951 | conforme | notebook v7, M0 |
| DSO4 inférence (valeur servie) | < 1 s | 0,440 s (L4), fiabilité active : 0,444 s ; 91 ms (A100, notebook) | conforme | `fiabilite_pcfno3d/.../session_20261016T134302Z/resume_dso4.json` |
| DSO5 RMSE à 25 % des capteurs | ≤ 0,7 m/s | 0,646 | conforme | notebook v7 |

**M1, dernière tentative bornée sur DSO1** (correction post-modèle du vent moyen par niveau,
règle pré-enregistrée C1 ET C2 ET C3) : mesure décisive 10i faite le 2026-10-16 (Colab
A100-SXM4-80GB, `notebooks/current/m1_bloc10i_colab.ipynb`) ; détail dans `docs/M1_PROTOCOLE.md`,
section « Résultat de la mesure décisive ».

| Condition | Mesure | Verdict |
|---|---|---|
| C1 test synoptique (120 instants, 50 % des capteurs) | pire niveau u 0,665 → 0,419 [IC 95 % 0,320 ; 0,559], v 0,508 → 0,382 [0,290 ; 0,496] ; R² min 0,9933 | conforme |
| C2 épisodes (46 instants jamais utilisés) | IC 95 % de Δ hors seuil : v à 150, 200 et 1000 m, u à 650 m ; pire niveau u 1,148 → 1,056, v 1,068 → 1,104 | **non conforme** |
| C3 | DSO2 2,58e-5 ; DSO3 0,951 ; DSO5 0,508 ; DSO4 121 ms (modèle seul 92 ms) | conforme |

- **Décision : correction NON RETENUE.** La v7 est **gelée seule** (tag `v7-fige`,
  `docs/resultats/modele_fige/README.md`). Le DSO1 reste non atteint à 650–1 000 m ; il est
  rapporté avec le critère OMM (DSO1 révisé proposé, conforme).
- Réserve imprimée par 10i : coefficients appris sur synthétique ; sur réel, l'erreur des
  baromètres (±10 à 50 Pa) est du même ordre que le signal géostrophique (26 Pa pour 10 m/s).
- Pour information, hors règle : à 100 % des capteurs, u 0,786 → 0,508, v 0,556 → 0,426. Jeu
  vierge (graine 2027) non mesuré (absent du Drive).
- À investiguer, sans effet sur le verdict : le `test_summary.json` du notebook v7 sur le Drive
  n'a plus le sha256 publié (modifié avant la session M1).

## 2. Fiabilité (M2, DSO6)

L'incertitude est un ensemble de 8 passes masquées, calibré sur la validation. L'intervalle à 80 %
vaut p50 = vitesse servie, avec p10 et p90 = vitesse ∓ 1,2816 σ.

| Mesure | Couverture 80 % (u / v / w / vitesse) | Spearman \|erreur\|, σ | Verdict |
|---|---|---|---|
| 1re lecture, 15/10, q = 0,8 | 0,751 / 0,811 / 0,794 / 0,796 | 0,289 / 0,307 / 0,229 / 0,290 | **non conforme** |
| 2e lecture, 16/10, q choisi sur la validation (règle croisée, 0,90) | 0,871 / 0,912 / 0,896 / 0,904 | 0,295 / 0,311 / 0,231 / 0,291 | **conforme (seconde lecture)** |

La règle de marge a été fixée **après** la première lecture du test : le second verdict n'est
donc pas une mesure indépendante. Les deux mesures sont publiées (`docs/FIABILITE.md`).

DSO4 et fiabilité :

- La fiabilité synchrone prend 4,2 à 4,7 s par instant, quelle que soit la taille de lot : une
  passe v7 sature déjà le L4.
- Sur décision de l'utilisatrice, la fiabilité est **différée**. DSO4 juge la valeur seule :
  **0,440 s, conforme** (0,444 s avec la fiabilité active). L'incertitude arrive **3,8 s** plus
  tard (lot de 4, pour information, sans verdict) ; l'API affiche « en_cours » puis « prete ».

## 3. Alertes (M4, AN1)

| Mesure | Dérives capteur détectées | Fausses alertes | Dérive des entrées | Verdict |
|---|---|---|---|---|
| v1, 16/10, test (protocole officiel 40 dérives + 10 pannes par jour) | 5,2 % | 0 % | détection 100 %, fausses alertes 6,7 % | **non conforme** |
| v2, 16/10, test, même protocole (cohérence avec les voisins + modèle de sites + persistance) | **100 %** | **2,2 %** | détection 100 %, fausses alertes 0 % | **conforme (seconde lecture)** |

Les pannes (valeurs figées) sont détectées à 100 %, hors verdict.

- **Détecteur v2 retenu sur la validation :** spatial, rayon 2, persistance sur 4 instants,
  correction de site, κ 10,5.
- **Détection par type de défaut (test) :**
  - biais : 0,5 m/s → 12 % ; 1 m/s → 97 % ; 2 et 3 m/s → 100 % ;
  - gain : 0,15 → 68 % ; 0,3 → 87 % ;
  - girouette : 15° → 83 % ; 30° → 100 %.
- **Dérive des entrées :** la règle v1 donnait 6,7 % de fausses alertes, dues à w (vent vertical) ;
  le vent est exclu en v2.
- **Statut :** la v2 a été conçue après la première lecture du test, donc son verdict est une
  seconde lecture.
- **Sources :** `alertes_pcfno3d/.../session_20261016T121344Z` (v1) et
  `session_v2_20261016T142959Z` (v2).

## 4. Éolien (M5, EOL1)

Mesure sur le parc **de démonstration** (3 éoliennes d'exemple, 12 MW), 120 instants de test :

- écart de production du parc entre la reconstruction et la vérité : **1,71 %** (seuil 5 %) ;
- nMAE de la puissance : 1,2 % ;
- RMSE du vent au moyeu : 0,32 à 0,43 m/s ;
- accord des états de fonctionnement : 98,9 %.

**EOL1 est conforme sur ce parc d'exemple.** Le verdict officiel demande la déclaration du parc
réel par Tukhnanutha.

## 5. Publication (M6, HIS1, DE3)

- **Rejeu :** les 120 instants de test sont publiés « comme en direct » au format du contrat
  rév. 2, avec fiabilité calibrée, alertes M4 et parc M5, en 9,1 s par instant (L4). Session
  du 16/10.
- **Atomicité (HIS1) :** testée contre le Worker simulé, avec coupures et panne après
  `_commit.json`. Le Worker réel a été corrigé (commit rejoué idempotent) et le client relit
  `/v1/times` et `/v1/latest` après chaque commit.
- **Exports NetCDF et CSV (spec 8.4) :** produits depuis l'instantané publié et vérifiés par
  schéma.

## 6. Preuves (M8, L6, L7, L9)

Comparaison avec les méthodes de référence, sur les mêmes instants, capteurs et nœuds que DSO1 :

| Méthode | RMSE max u, v | RMSE global u / v | R² min |
|---|---|---|---|
| **PC-FNO3D v7** | **0,665** | **0,542 / 0,437** | **0,991** |
| loi de puissance + IDW | 5,027 | 2,974 / 3,474 | 0,569 |
| loi de puissance + krigeage | 5,031 | 2,979 / 3,478 | 0,568 |
| profil log + IDW | 6,481 | 4,171 / 3,269 | 0,156 |
| profil log + krigeage | 6,487 | 4,176 / 3,275 | 0,155 |

Le variogramme (exponentiel, pépite ≈ 0, palier 1,10, portée 7,7 km) et α (0,06) sont ajustés sur
la validation. Session : `preuves_pcfno3d/best_dso_v7/session_20261016T125440Z`.

| Étude | Résultat mesuré (test, 120 instants) | Lecture |
|---|---|---|
| E1 densité du réseau | PC-FNO3D tient DSO5 (≤ 0,7 m/s) jusqu'à **50 % des capteurs** ; aucun réseau régulier testé (un capteur tous les 300 m, 500 m ou 1 km) ne le tient ; aucune référence ne le tient, quelle que soit la densité | densité recommandée, au moins la moitié du treillis actuel |
| E2 placement guidé par l'incertitude | RMSE max 2,39 m/s (placement « optimal ») contre 1,28 (hasard moyen, σ 0,02) : gain **négatif**, −1,11 m/s | **résultat négatif** : le σ M2 ne guide pas le placement |
| E3 capteur virtuel de secours | erreur vectorielle RMS : krigeage 0,19, IDW 0,27, plus proche voisin 0,45, PC-FNO3D 1,50 m/s | sur ce treillis très dense, les voisins à 100 m valent mieux que la reconstruction 3D |
| E5 sobriété | inférence : 9,3 µWh GPU par reconstruction (0,47 s, nvidia-smi, GPU seul) ; CO₂ non calculé (intensité carbone non fournie) ; entraînement non mesuré | à compléter : intensité carbone, mesure de l'entraînement, référence WRF à sourcer |

La data card, la model card et les diagrammes UML sont dans `docs/`.

## 7. Données, API, tableau de bord

| Volet | Critère | Résultat |
|---|---|---|
| Contrat de données ANEO (M3) | DE1 | conforme |
| Reproductibilité (M0) | DE2 | conforme |
| API et MCP (M7) | API1, MCP1 | conforme : fermés par défaut, révocation, quotas, journal |

Tableau de bord ATOM, lots F0 à F6 :

- contrat v1 rév. 2 ;
- front Vite / React / three.js ;
- Worker avec R2 ;
- scène 3D : relief, lignes de courant, hauteur au-dessus du sol ;
- pages Vue d'ensemble, Fiabilité, Modèle, Éolien, Alertes et Comparaison ;
- export CSV ;
- charte graphique ;
- scénario de démonstration.

La CI est verte. Le déploiement est **local** sur ines-pc, par décision de l'utilisatrice.

## 8. Ce qui reste ouvert

- **DSO1 à 650–1 000 m :** non atteint. M1 a rejeté la correction post-modèle (C2 non
  conforme) ; la v7 est gelée seule et la limite est assumée et documentée, avec le critère
  OMM. Pas de v8 ni de réentraînement sans accord de Rayen.
- **Gel de la v7 :** restent à faire sur le Drive l'archive `fige/modele_fige_2026-10.zip`, la
  copie de `decision_10i.json` dans le dépôt et l'enquête sur le sha256 de `test_summary.json`
  (`docs/resultats/modele_fige/README.md`).
- **EOL1 officiel :** il faut le parc réel.
- **Données réelles :** aucun capteur ANEO réel n'est encore importé, et l'ancrage géographique
  du domaine reste supposé.
