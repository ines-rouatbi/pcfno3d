# ATOM Wind — interface éditoriale

## Périmètre et direction

Le propriétaire a choisi une interface **claire, éditoriale, aérée, avec des graphiques forts**.
La scène 3D sera reprise séparément dans Pro Tuk-ai : la présente livraison se concentre sur
le tableau de bord, les autres écrans, le socle React/Tailwind et leur validation.

L'interface initiale concentrait navigation, unités, thèmes, langues, recherche, compte, lecture
horaire et source dans un en-tête à plusieurs rangées. Les huit écrans et leurs contrôleurs
étaient réunis dans un HTML d'environ 185 Ko. Les indicateurs, recommandations et éléments
secondaires partageaient presque tous le même traitement en cartes, ce qui affaiblissait
la hiérarchie. Le laboratoire utilisait une seconde feuille de style et une seconde navigation.

Le tableau de bord organise le travail en trois étapes : situer le vent sur le territoire,
choisir un lieu et ouvrir l'outil utile à son activité. La navigation latérale sépare territoire,
activités, compréhension des données et gestion. Elle indique la vue active et respecte les rôles.
Sur mobile et tablette étroite, elle devient un panneau avec focus contenu et retour au bouton
d'ouverture à la fermeture. Les boutons Précédent/Suivant du navigateur retracent les changements
de vue, y compris depuis la recherche.

La carte 2D utilise la projection géographique et le même échantillonneur de vent que les
instruments existants. Les lieux se sélectionnent sur la carte, au clavier, dans la liste ou le
menu local. Le panneau adjacent distingue vent local et moyenne territoriale ; il présente
rafales, direction et fiabilité indicative. Le lieu sélectionné est repris dans Mon vent et,
depuis le raccourci Agriculture, dans le choix de parcelle. Les alertes du briefing ouvrent
leur détail sélectionné. Le graphique quotidien reste régional : tiges pour les rafales maximales,
trace bleue pour le vent moyen, bande ocre pour l'heure choisie. La couverture, le réseau et la
validation se consultent à la demande. Les commandes horaires n'apparaissent pas en Administration.

### Références étudiées

- [NN/G : AI Prototyping in Real Design Contexts](https://www.nngroup.com/articles/ai-prototyping/)
  identifie notamment les défauts de regroupement, de hiérarchie et les interfaces génériques
  issues de bibliothèques de composants. L'audit de notre propre interface a retrouvé ces défauts :
  titres promotionnels, numérotation de sections, indicateurs répétés et textes secondaires trop petits.
- [Pentagram : Climate Tech Map](https://www.pentagram.com/work/climate-tech-map)
  montre un langage graphique construit autour de la structure des informations. La référence
  retenue est cette méthode ; ni sa roue, ni ses illustrations, ni ses éléments de marque ne sont copiés.
- [GAIN : Met Office](https://www.thisisgain.com/work/met-office-designing-an-ambitious-new-app-and-digital-language/)
  décrit un travail organisé autour des usages météo, des parcours et de l'accessibilité.
- [Giorgia Lupi : Data Humanism](https://giorgialupi.com/data-humanism-my-manifesto-for-a-new-data-wold)
  souligne le rôle du contexte dans la communication des données. Ici, la lecture indique le lieu,
  l'heure, le sens du vent, les rafales et l'origine du calcul.
- [ECMWF : présentation des cartes](https://sites.ecmwf.int/www/addons/support/charts-tour.html)
  décrit le passage d'une carte à la lecture d'un lieu. Cette liaison carte → conditions locales
  a guidé le parcours, sans reprendre la présentation visuelle d'ECMWF.

Les pages de référence ont été consultées et certaines ont été capturées pour l'étude.
Aucun média externe de ces références n'est utilisé dans l'application.

## Système visuel

- IBM Plex Serif pour la marque et les titres, Manrope pour les textes et la lecture principale,
  IBM Plex Mono pour les repères techniques.
  Les polices sont servies localement, sans requête Google Fonts.
- Fond clair, encre bleu sombre, bleu de mer pour les commandes et le vent moyen,
  brun chaud pour les tiges des rafales et ocre pour l'heure choisie. La carte associe
  terre claire, mer bleue et contours schématiques ; le panneau local est bleu profond.
  Les couleurs de vitesse du champ et les couleurs de décision conservent leurs conventions.
- Espacements construits autour de 4, 8, 12, 16, 24, 32 et 48 px. Les indicateurs secondaires
  sont séparés par des filets plutôt que par des cartes indépendantes.
- Boutons, champs, pastilles, listes, tableaux et focus partagent les mêmes variables.
  Le thème sombre reste disponible dans les préférences.
- Sur mobile : navigation sur demande, sections empilées, tableaux défilants dans leur conteneur,
  fiche de point en panneau inférieur et contrôles tactiles lisibles.
- Transitions courtes ; `prefers-reduced-motion` coupe les animations. Les dialogues de
  connexion, visite et recherche conservent le focus au clavier.

## Code et responsabilité

| Élément | Source |
| --- | --- |
| React 19.3, TypeScript, Vite 8.3, Tailwind 4.3 | `package.json`, `vite.config.ts`, `tsconfig.json` |
| Navigation, contexte, préférences, assistant | `frontend/src/components.tsx` |
| Démarrage et notifications de présentation | `frontend/src/workspace.ts` |
| Écrans React | `frontend/src/views/` |
| Tokens | `frontend/src/styles/tokens.css` |
| Navigation et composition du tableau de bord | `frontend/src/styles/workspace-layout.css` |
| Instruments et éléments communs | `frontend/src/styles/atmos.css` |
| Vue géographique 2D accessible | `frontend/src/territory-map.ts` |
| Compatibilité des instruments existants | `frontend/src/styles/legacy.css` |
| Styles de la scène confiée à Pro Tuk-ai | `frontend/src/styles/scene-ui.css` |
| Terrain Tukai fourni, streaming et intégration géographique | `frontend/src/terrain/`, `docs/TERRAIN_INTEGRATION.md` |
| Contrôleurs, données, calculs de présentation conservés | `frontend/src/engine/` |
| Laboratoire (outil interne) et visualiseur d'export | `science.tsx`, `export.tsx`, `styles/science.css` |
| Traductions | `frontend/public/atom-wind/i18n/` |
| Tests de navigateur | `frontend/tests/workspace.spec.ts`, `frontend/tests/dashboard-flow.spec.ts` |

Les vues React sont mémorisées pour que leurs conteneurs d'instruments restent stables.
Les contrôleurs scientifiques existants possèdent leurs canvas, SVG, tableaux et formulaires.
Le shell lit des notifications `atom:state` ; il ne recalcule pas le vent ou les permissions.
Ce choix conserve les contrats existants et évite une réécriture des calculs pendant la migration.
Une migration ultérieure des contrôleurs impératifs vers des composants plus fins reste possible.

## Laboratoire : outil interne

Le laboratoire (`/app/`) est l'ancienne interface de développement du package : coupe brute à une
altitude ASL, profil en coordonnées locales, verdict DSO archivé, exemples d'intégration MCP/API.
ATOM Wind couvre tout cela pour un client (carte et vue 3D, fiche point, qualité via
`/api/atom-wind/qualite`). Il ne fait donc plus partie du parcours SaaS : aucun lien depuis
ATOM Wind, et la racine `/` redirige vers `/app/atom-wind/`. Il reste servi pour l'équipe
(diagnostic d'un modèle chargé) et le visualiseur d'export y renvoie.

## Construire et lancer

```powershell
npm ci
npm run build
# Dans un second terminal, lancer le service selon la configuration existante.
python -m pcfno3d serve --port 8000
```

- Application : `http://127.0.0.1:8000/app/atom-wind/`
- Laboratoire (outil interne, aucun lien depuis ATOM Wind) : `http://127.0.0.1:8000/app/`
- Export : `http://127.0.0.1:8000/app/export/?src=/api/export/STEM`
- Développement : `npm run dev`, puis `http://127.0.0.1:5173/app/atom-wind/`.
  Vite relaie `/api` et `/mcp` au service local sur le port 8000.

Le serveur conserve ses exigences de clé et de rôle. Les captures des espaces restreints
utilisent le compte de démonstration administrateur existant, sans modifier les droits du service.

`npm run build` produit les fichiers servis dans `webapp/` et l'entrée autonome `web/index.html`.
Il faut construire avant de créer une archive Colab. Le Dockerfile construit lui-même le frontend
dans une étape Node, puis copie les fichiers dans l'image Python. Le script historique
`python scripts/build_atom_wind.py` appelle désormais cette construction ; l'ancien prototype
ne peut plus écraser accidentellement l'entrée React.

Après modification des textes :

```powershell
node scripts/extract_frontend_markup.mjs
python scripts/extraire_textes_atom_wind.py
npm run build
```

L'extraction utilise le rendu statique des composants React et les contrôleurs sources,
jamais les bundles minifiés. Les dictionnaires existants et leurs traductions restent conservés.

## Validation et captures

```powershell
npm run test:browser
python -m pytest tests/test_atom_wind.py tests/test_webapp_and_mcp.py tests/test_securite_api.py tests/test_alerts.py tests/test_windfarm.py --tb=short
```

Les tests de navigateur attendent un service local disposant du mini-dataset et de son export
`006_000000`. Ils vérifient les huit écrans à 1440 et 390 px, le changement de lieu et d'activité,
les unités, les thèmes, l'heure, la recherche clavier, les profils, les droits, les alertes locales,
l'agriculture, l'éolien, le laboratoire, les langues et le visualiseur d'export. La vue « Vent sur le
terrain » intègre désormais l'export Tukai fourni et les prévisions nationales ; voir
`TERRAIN_INTEGRATION.md` pour ses sources, ses limites et ses tests propres.
Les parcours ajoutés vérifient le lieu carte → Mon vent → parcelle, le retour navigateur,
l'ouverture de l'alerte correspondante, le focus du panneau de navigation et les largeurs
intermédiaires 768, 900, 1024 et 1280 px.

Captures et résultats sont dans `output/redesign/`. Le tableau de bord final est
`1440-tableau.png` sur ordinateur et `390-tableau.png` sur mobile. Les JSON contiennent
les résultats de navigateur et les mesures effectuées, sans annoncer une performance matérielle
générale à partir d'un navigateur sans interface.
La première fenêtre visible est dans `dashboard-first-screen-final.png` ; le thème sombre,
dans `dashboard-dark-final.png`. `node scripts/capture_dashboard.mjs -final` reproduit ces captures.

Résultats du 4 novembre 2026 sur la composition et les parcours actuels :

- Construction TypeScript/Vite réussie.
- 30 tests Python ciblés réussis (`test_atom_wind.py`, `test_webapp_and_mcp.py`).
  La précédente validation plus large de 114 tests est conservée dans
  `output/redesign/python-results.xml` ; aucun calcul scientifique n'a changé depuis.
- Les résultats de navigateur actuels sont enregistrés dans `output/redesign/browser-results.json`.
  La suite comprend également le terrain réel, le vent animé, la couverture nationale et Leaflet.
- Les 9 tests de service Colab avaient également réussi lors de la validation de la migration.
- Mesure locale Chrome sans interface, 1440 × 1050, HTTP local, journée synthétique : première
  notification de disponibilité à 321 ms, LCP à 476 ms, CLS 0,071. Les sept changements d'heure
  mesurés prennent 9,1 ms en médiane et 58,4 ms au maximum jusqu'à la frame suivante.
- Le module Three.js n'est pas téléchargé lors de l'ouverture du tableau de bord.
  Ces chiffres décrivent le poste et le mini-jeu testés ; ils ne constituent pas une mesure de production.
- L'archive de code avec les bundles est produite dans `output/redesign/pcfno3d-code.zip`.

## Limites explicites

- Le service local disponible sert un mini-jeu synthétique 4 × 2 pour ATOM, et non le domaine
  de production Grand Tunis 136 × 112. ATOM affiche donc une démonstration clairement signalée ;
  le laboratoire continue de lire le mini-jeu via les véritables API. Le domaine de production,
  les sorties PC-FNO3D opérationnelles et l'assistant connecté nécessitent un service approprié.
  Les prévisions nationales GFS de la vue terrain ont une source et un lecteur distincts ; elles
  ne sont pas des sorties PC-FNO3D et ne rendent pas les outils locaux disponibles hors Grand Tunis.
- Le frontend refuse de présenter cette petite grille comme une carte détaillée du Grand Tunis.
  Relief schématique, simulation, interpolation et fiabilité indicative sont signalés.
- Les valeurs historiques de validation présentes dans le prototype restent des archives ;
  `/atom-wind/qualite` remplace le tableau de qualité lorsqu'un modèle de production est disponible.
- Les tests de comptes/routes utilisant `forcage_era5.npz` échouent dans la fixture scientifique
  de cet environnement : absence d'échéance à moins de trois heures du 02/01/2025 à 12 h UTC.
  La même erreur a été reproduite sur le commit inchangé `36cedc9` dans un worktree séparé.
  Les calculs scientifiques n'ont pas été modifiés pour contourner ce problème.
- Les graphiques dessinés dans un canvas conservent leurs textes français, comme dans l'interface
  d'origine. Les nouveaux textes du shell et du tableau de bord ont des traductions anglaises/arabes.
- La documentation interactive FastAPI conserve son interface standard et ses contrats.
- Le terrain Tukai fourni le 4 novembre est intégré dans le premier mode de la vue 3D.
  Le mode d'analyse conserve les outils et le relief du champ de vent dans `atom-3d.js`
  et `init3d()`. Leurs coordonnées et leurs données ne sont pas remplacées par le terrain national.
- Le fond de carte charge Leaflet depuis les fichiers de l'application. Les tuiles de rues
  OpenStreetMap conservent leur dépendance réseau ; le relief reste utilisable sans elles.
