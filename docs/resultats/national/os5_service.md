# OS5 — ATOM Wind sert le national à côté du Grand Tunis (B5–B7)

Run du 2026-11-15, 15:47–16:46 (heure de Tunis), Colab **L4**, carnet
`notebooks/current/pcfno3d_atom_deux_domaines_colab.ipynb` (code `fd48cc1`, correctif de copie
`e8c85c6`). La session a été terminée à la fin (`stop(info)` puis `runtime.unassign()`).

## B5 — DSO4 sur le GPU du service (médiane de 20 passes après 3 de chauffe)

| Modèle servi | GPU | Médiane | p95 | Max | Mémoire max | Seuil | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| National C' s44 (`national_v4_service`) | L4 | 1,010 s | 1,028 s | 1,028 s | 9,73 Go | 1 s | **non conforme** |
| Grand Tunis v4 (`grand_tunis_v2_service_colab`) | L4 | 1,118 s | 1,123 s | 1,123 s | 10,44 Go | 1 s | **non conforme** |

Fichier : `MyDrive/pcfno3d_national/os5/dso4_service.json`. Sur A100 (carnet du test 2022), le
même modèle national mesurait 264 ms : DSO4 dépend du GPU. Sur L4, le national dépasse le seuil de
10 ms. Pour un DSO4 conforme en service, il faut servir sur A100 (coût en unités Colab plus élevé :
décision d'Ines).

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

Cause trouvée après le run : `heureLocale()` lisait `Intl.DateTimeFormat("fr-FR", {hour: "numeric"})`,
qui renvoie « 16 h » ; `Number("16 h")` vaut NaN, donc l'heure de la page devenait NaN en temps réel
(Grand Tunis compris). Corrigé dans `d7a7c0b` (lecture de la seule partie heure). Le second défaut
en découle très probablement (réponse calculée à l'heure NaN) : à vérifier au prochain run. Le
carnet (`3d28355`) construit maintenant le front-end et refait les captures.

Adresse publique : atom.zelqane.com sert encore l'ancien front-end du Worker (déploiement de Rayen) ;
la page à deux domaines est servie à `atom-origin.zelqane.com/app/atom-wind/?domaine=national`
quand le carnet tourne.
