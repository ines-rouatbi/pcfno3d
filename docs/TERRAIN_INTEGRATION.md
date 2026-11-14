# Terrain Tunisie — intégration ATOM Wind

Le dossier fourni `ATOM_Wind_Tunisia_Terrain` est importé sans modifier le projet Tukai,
ses sources ni ses maillages. L'application utilise les GLB compressés, les grilles de hauteur
et les couches d'eau du sous-dossier `ATOM_Wind_Tunisia_Web`.

## Import et actualisation des prévisions

```powershell
node scripts/import_tukai_terrain.mjs "C:\Users\mahdi\Desktop\tukai-pro\artifacts\ATOM_Wind_Tunisia_Terrain"
python scripts/fetch_national_wind.py
npm run build
```

L'import copie 5 272 fichiers, environ 458,3 MiB, vers `frontend/public/terrain/tunisia`.
Vite les sert en développement et les copie vers `webapp/terrain/tunisia` pour le service Python.
Ces dossiers générés sont ignorés par Git : une nouvelle installation doit exécuter l'import.
Les sources géographiques brutes et les intermédiaires ne sont pas recopiés. L'archive Colab
transporte le terrain dans `webapp` une seule fois.

## Servi par le service, pas par le build

Le relief est une **donnée du service**, comme les poids du modèle : il n'est ni dans git ni
dans l'image Docker (`.dockerignore`). Le service le monte sous `/app/terrain/` depuis, dans
l'ordre :

1. `PCFNO3D_TERRAIN_DIR` (dossier qui contient `tunisia/`) — le cas de production : un volume
   (`docker run -v /chemin/vers/terrain:/terrain:ro`, l'image définit déjà
   `PCFNO3D_TERRAIN_DIR=/terrain`) ;
2. `webapp/terrain` (la copie faite par `npm run build`) ;
3. `frontend/public/terrain` (la sortie de l'import).

Les fichiers partent avec `Cache-Control: public, max-age=86400`. Sans relief, `/app/terrain/…`
répond 404 avec la commande d'installation, `/api/health` indique `"terrain_3d": false`, et la
Carte 3D affiche « Relief 3D non installé sur ce serveur » ; les autres vues fonctionnent.
Le dossier source `ATOM_Wind_Tunisia_Terrain` n'est publié nulle part : il se transmet à part.

Application : `http://127.0.0.1:5173/app/atom-wind/#carte3d` en développement,
ou `http://127.0.0.1:8000/app/atom-wind/#carte3d` avec le service Python.

## Sources de vent

« Vent sur le terrain » ouvre par défaut **Tunisie · GFS**, avec des prévisions nationales
NOAA GFS fournies par [Open-Meteo](https://open-meteo.com/en/docs/gfs-api).
L'instantané contient 170 positions sur une grille de lecture de 0,5°, 48 heures et les niveaux
10 / 80 m du sol. Cette grille de lecture n'est pas une résolution de modèle annoncée.
Les positions des cellules retournées sont conservées. Les composantes est/nord sont interpolées
pour l'affichage ; les directions ne sont pas interpolées directement autour de 0 / 360°.
Le terrain détaillé ne transforme pas cette prévision en calcul atmosphérique local à 32 m.

Le lecteur indique la date prévue, l'heure de Tunis, la récupération et l'attribution.
Un instantané local ou un cache de moins d'une heure est utilisé s'il contient encore l'heure
actuelle ; sinon, le navigateur tente le fournisseur. Après une panne, seul un instantané réel
daté de moins de six heures et contenant encore l'heure actuelle peut servir de cache signalé.
Sans données utilisables, la vue indique l'indisponibilité et propose Réessayer ; aucun vent
synthétique ne remplace une prévision nationale. Une installation durable doit automatiser
l'actualisation de l'instantané selon les conditions du fournisseur.

La source locale conserve le domaine PC-FNO3D existant, **68 × 56 km dans le Grand Tunis**,
et ses outils. L'adaptateur longitude/latitude ne change pas les calculs scientifiques.
Si le service de production manque, le champ synthétique est explicitement marqué démonstration.
Un point national dans le Grand Tunis peut ouvrir les conditions locales via « Comparer au champ
local » ; hors de cette zone, la prévision nationale reste lisible sans inventer de calcul local.

Les boutons Tunisie / Grand Tunis déplacent la caméra. Le sélecteur Source change les données.
Les indications de territoire suivent la source et le mode : Tunisie pour les prévisions
nationales, Grand Tunis pour la source locale et les outils d'analyse avancés.

## Relief réel des outils d'analyse avancés

Les outils d'analyse (scène 3D du domaine 68 × 56 km) et les cartes 2D de la page lisent
`frontend/public/atom-wind/relief/grand-tunis.{json,bin.gz}` : une grille de 125 m (544 × 448)
tirée de ce terrain — altitude moyenne de 4 × 4 points DEM par cellule, mer et lacs rasterisés
depuis `water/mediterranean.glb` et `water/natural-lakes.glb` — sur la même projection linéaire
que la page (`LAT0`, `LON0`, `KX`, `KY`). Ce fichier (≈ 330 Ko) est versionné, contrairement au
terrain : il se régénère après un nouvel import.

```powershell
node scripts/build_grand_tunis_relief.mjs
```

- Avec le service PC-FNO3D, la grille de la page reste celle du modèle (`/api/atom-wind/grille`) ;
  la grille fine ne sert alors qu'au maillage affiché.
- Sans service, elle remplace le relief schématique de la maquette ; le vent reste une
  démonstration synthétique, et la bannière le dit (« relief réel du Grand Tunis »).
- Sans le fichier, la page revient au relief schématique.
- Les couleurs de la scène se déduisent de l'altitude et de la pente (champs, garrigue, forêt,
  roche) : ce n'est pas une carte d'occupation du sol. La teinte urbaine n'apparaît qu'avec la
  rugosité réelle servie par le modèle.

## Coordonnées et altitudes

- Géométrie : EPSG:32632, origine E=500 000 m / N=3 800 000 m. X vers l'est de grille,
  Y vers le haut, Z vers le sud. Les translations GLB ne sont pas appliquées deux fois.
- Altitudes : mètres orthométriques EGM2008. Exagération verticale 1. Les fonctions UTM
  et le décodeur de hauteur fournis sont conservés dans `frontend/src/terrain/source`.
- Les marqueurs interrogent les grilles DEM indépendantes. Sans échantillons valides,
  l'altitude du maillage peut être affichée comme « Relief affiché ». Une altitude GPS
  ellipsoïdale n'est pas présentée comme une hauteur DEM.
- Les composantes du vent sont projetées en UTM ; les hauteurs AGL sont ajoutées au terrain.
  Les traînées sont illustratives et leur mouvement accéléré, comme indiqué dans la légende.
- Le lecteur gzip reconnaît aussi les réponses déjà décompressées par HTTP, notamment
  celles de Vite en développement : les grilles ne sont pas décompressées deux fois.

## Chargement et ressources

L'aperçu national, les deux GLB d'eau et les manifestes sont chargés au démarrage. La couche
nationale construit sa surface d'affichage à partir de cet aperçu et de l'eau ; elle ne télécharge
pas toutes les grilles détaillées du pays. La source locale prépare les grilles du Grand Tunis.

Deux secteurs régionaux au maximum sont affinés sur ordinateur, un sur téléphone ; une seule
zone locale est affinée davantage. Les lots comptent quatre téléchargements parallèles.
Un parent reste visible jusqu'à ce que tous ses enfants soient prêts, puis est remplacé
atomiquement. Les requêtes périmées sont annulées et le détail est libéré au retour national.
Le mode Économe évite le niveau local.

Le vent est activé par défaut et animé dans le canvas du terrain. Pause et les préférences
de mouvement réduit permettent de garder les traînées statiques. Une vue cachée ne dessine plus.
Caméra, sélection, heure et hauteur actualisent aussi la scène. Souris, tactile et clavier
sont disponibles. Après une erreur, Réessayer ne duplique pas le canvas.

Les niveaux 512 / 128 / 32 m décrivent les grilles de construction, pas une précision garantie.
Copernicus GLO-30 est un modèle de surface ; végétation et constructions peuvent influencer
les hauteurs. WorldCover 2021 n'est pas une mesure météo. L'eau n'est pas une bathymétrie.

## Fond de carte et validation

Leaflet est servi localement depuis `/app/vendor/leaflet/leaflet.js`, avec ses images et sa licence.
`scripts/copy_frontend_vendor.mjs` prépare ces fichiers avant `dev` et `build`. Le chargement
partage une promesse entre clics simultanés et peut être réessayé. Un échec conserve le relief.
Les rues OpenStreetMap nécessitent séparément l'accès à leurs tuiles ; une panne des tuiles
n'est pas présentée comme un échec de chargement de la bibliothèque.

Les attributions sont accessibles depuis « Sources du terrain et limites » et
`/app/terrain/tunisia/LICENSES.md` : Copernicus, ESA WorldCover, OpenStreetMap/geoBoundaries,
Natural Earth et GeoNames. Les prévisions sont attribuées à NOAA GFS / Open-Meteo, CC BY 4.0.

`frontend/tests/terrain.spec.ts` vérifie les assets réels, l'exclusivité des niveaux, les budgets
de résidence, le mouvement visible des pixels, la pause, HTTP gzip, le téléphone, les prévisions
à Sfax / Tozeur / Tataouine, les sources et les outils locaux. `map-loader.spec.ts` vérifie
la bibliothèque auto-hébergée, les clics simultanés, le relief de secours et le nouvel essai.
Les tests existants conservent la couverture des outils d'analyse historiques.

Captures : `output/redesign/terrain-national-forecast.png`, `terrain-national-mobile.png`,
`terrain-wind-overlay.png` et `terrain-mobile.png`.
