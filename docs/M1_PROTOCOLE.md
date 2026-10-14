# M1 — Protocole de la dernière tentative DSO1 (650–1000 m)

*Document en cours de rédaction. Cette première version ne fige que la règle de décision,
avant toute mesure ; le reste du protocole suit dans des commits ultérieurs, sans modifier
la section ci-dessous.*

## Règle de décision pré-enregistrée (validée par Rayen le 2026-10-14, salon #325)

**Objet.** Décider si la correction post-modèle de la v7 (estimateur géostrophique v2 + ridge
par niveau prédisant le décalage moyen du niveau) est retenue. La règle est écrite et versionnée
**avant toute mesure** ; elle ne sera pas modifiée après coup.

- **C1 — test synoptique** (protocole DSO1, 120 instants, 50 % des capteurs) : pire niveau
  évalué u ET v après correction ≤ 0,5 m/s. IC 95 % bootstrap sur les instants (1000 tirages)
  RAPPORTÉ, borne haute publiée mais non utilisée comme seuil (le test a servi à la conception
  de la correction).
- **C2 — épisodes** (46 instants jamais utilisés pour le choix), Δ = RMSE après − RMSE avant,
  bootstrap APPARIÉ sur les instants (1000 tirages), pour u et v. Sur chaque niveau 650, 800
  et 1000 m : borne haute de l'IC 95 % de Δ ≤ 0 (amélioration significative). Sur chaque
  niveau évalué de 100 à 500 m : borne haute de l'IC 95 % de Δ ≤ +0,02 m/s (pas de
  dégradation notable).
- **C3** — DSO2 (moyenne ≤ 1e-4), DSO3 (rugosité dans [0,9 ; 1,1]), DSO5 (RMSE ≤ 0,7 à 25 %
  des capteurs) et DSO4 (temps total modèle + correction < 1 s par instant, GPU) restent
  conformes.
- **Pour information, hors règle** : cas 100 % capteurs, valeurs absolues avant/après avec IC.
- **Décision** : correction RETENUE ⇔ C1 ET C2 ET C3. Sinon : gel de v7 seule, DSO1 rapporté
  honnêtement (critère OMM). Méthode figée : ridge par niveau 10h, alpha = 0,01 choisi sur la
  validation, paramètres dans couche_limite_v7.json (sha256 imprimé par la cellule 10i). Pas
  de v8 sans accord de Rayen.
- **Mesure** : cellule BLOC 10i du notebook Colab fno3d_v7.ipynb, la règle étant imprimée
  avant tout calcul.
