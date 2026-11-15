# OS5 — ATOM Wind sert le national à côté du Grand Tunis (B5–B7)

Run du 2026-11-15, 15:47–16:46 (heure de Tunis), Colab **L4**, carnet
`notebooks/current/pcfno3d_atom_deux_domaines_colab.ipynb` (code `fd48cc1`, correctif de copie
`e8c85c6`). La session a été terminée à la fin (`stop(info)` puis `runtime.unassign()`).

## B5 — DSO4 sur le GPU du service (médiane de 20 passes après 3 de chauffe)

| Modèle servi | GPU | Médiane | p95 | Max | Mémoire max | Seuil | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| National C' s44 (`national_v4_service`) | L4 | 1,010 s | 1,028 s | 1,028 s | 9,73 Go | 1 s | **non conforme** |
| Grand Tunis v4 (`grand_tunis_v2_service_colab`) | L4 | 1,118 s | 1,123 s | 1,123 s | 10,44 Go | 1 s | **non conforme** |

Sur L4, le national dépasse le seuil de 10 ms. **Second run sur A100 (15/11, 16:40–17:00, carnet
`3d28355`, option choisie par Ines) : conforme.**

| Modèle servi | GPU | Médiane | p95 | Max | Mémoire max | Verdict |
|---|---|---:|---:|---:|---:|---|
| National C' s44 | A100-SXM4-80GB | **0,228 s** | 0,228 s | 0,228 s | 9,73 Go | **conforme** |
| Grand Tunis v4 | A100-SXM4-80GB | **0,263 s** | 0,264 s | 0,264 s | 10,44 Go | **conforme** |

Fichier : `MyDrive/pcfno3d_national/os5/dso4_service.json` (le run A100 a remplacé celui du L4,
dont les chiffres sont gardés ci-dessus). DSO4 est donc tenu en service à condition de servir sur
A100.

## B7 — démonstration de bout en bout (API du service, aujourd'hui 12 h, temps réel)

| Ville | Domaine choisi par la position | Vent 10 m | Source 10 m | Vent 100 m | Source 100 m |
|---|---|---:|---|---:|---|
| Sfax | Tunisie | 2,81 m/s | GFS corrigée par les aéroports (OI des METAR) | 2,68 m/s | modèle national |
| Gafsa | Tunisie | 1,70 m/s | idem | 1,81 m/s | modèle national |
| Tozeur | Tunisie | 0,91 m/s | idem | 0,86 m/s | modèle national |
| Gabès | Tunisie | 1,23 m/s | idem | 0,89 m/s | modèle national |
| Tunis | Grand Tunis | 1,38 m/s | modèle Grand Tunis (PC-FNO3D) | 1,34 m/s | modèle Grand Tunis |

Fichier : `MyDrive/pcfno3d_national/os5/demo_points.json`. `/api/atom-wind/domaines` liste les deux
domaines (Grand Tunis principal, national avec sa couche 10 m). La clé de démonstration a été créée
puis révoquée sans être affichée.

## Page ATOM Wind (captures)

Captures par Chromium headless sur Colab (`scripts/captures_os5.py`, clé temporaire révoquée) :
`MyDrive/pcfno3d_national/os5/captures/` — `national_monvent_{sfax,gafsa,tozeur,gab}.png`,
`national_carte3d.png`, `grand_tunis_monvent_tunis.png`, `national_reseau_aneo.png`, `captures.json`.

Corrigé pendant le run :
- la page servie est le front-end React (`frontend/`) : il ne transmettait pas `?domaine=` au
  service et montrait toujours le Grand Tunis (`202894d`) ; l'en-tête affiche « Tunisie » sur toute
  la page nationale (`0ecd3f2`) ;
- les fichiers construits (`webapp/assets/`, ignorés par git) manquaient sur Colab : le carnet doit
  lancer `npm ci && npm run build` avant le service (à ajouter au carnet).

Défauts vus sur les captures du 15/11 :
- le curseur d'heure affiche « NaN h » en temps réel ;
- la carte « point sélectionné » et la réponse restent sur Tunis-Carthage quand on choisit une autre
  ville (Sfax…) dans « Où êtes-vous ? ».

Cause : `heureLocale()` lisait `Intl.DateTimeFormat("fr-FR", {hour: "numeric"})`, qui renvoie
« 16 h » ; `Number("16 h")` vaut NaN, donc l'heure de la page devenait NaN en temps réel (Grand Tunis
compris). Corrigé dans `d7a7c0b`.

**Vérifié au run A100** (captures dans `docs/demo/`) : le curseur affiche « 15 h », le champ de vent
national s'affiche sur la carte de la Tunisie, et le choix de la ville met à jour le point et la
réponse (Gafsa 11 km/h, Tozeur 6 km/h, Gabès 16 km/h « le produit peut dériver », Tunis-Carthage
7 km/h sur le Grand Tunis). Ces captures montrent la journée de test 01/07/2022 (la page a démarré sur
le jeu de test) et non le temps réel.

Limites restantes, constatées sur les captures :
- Sfax : la première capture a été prise avant que la liste des lieux soit prête (« Remada » affiché),
  elle n'est pas gardée ;
- onglet « Réseau ANEO » : réservé à un compte administrateur du SERVICE ; la session de
  démonstration locale de la page ne suffit pas (« plan d'exemple ») ;
- carte 3D : le relief ne se charge pas dans Chromium headless (WebGL logiciel) : à capturer depuis
  un vrai navigateur.

`docs/demo/` : `os5_national_gafsa.png`, `os5_national_tozeur.png`, `os5_national_gabes.png`,
`os5_grand_tunis_tunis.png` (copies d'écran de l'aperçu Drive des PNG produits sur Colab ; les
originaux 1440 × 900 sont dans `MyDrive/pcfno3d_national/os5/captures/`).

Adresse publique : atom.zelqane.com sert encore l'ancien front-end du Worker (déploiement de Rayen) ;
la page à deux domaines est servie à `atom-origin.zelqane.com/app/atom-wind/?domaine=national`
quand le carnet tourne.
