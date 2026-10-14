# Reproductibilité (DE2)

> Exigence : « même entrée et même version donnent la même sortie (tolérance documentée) ».

## Ce qui est garanti

Une reconstruction (`WindFieldModel.predict`, `predict_stem`, `predict_csv`, donc aussi
l'API `/api/reconstruct*`, le MCP et `pcfno3d reconstruct`) est **identique bit à bit**
(écart maximal **0,0 m/s** sur u, v, w) lorsque tout ce qui suit est identique :

| condition | comment la vérifier |
|---|---|
| exécution sur **CPU** | `--device cpu` ; c'est le cas du service local |
| même version de **torch** (et de numpy) | `python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"` |
| même **checkpoint** | empreinte sha256 du `.pt` (v7 : `b2529442…`, affichée par `scripts/install_v7_weights.py`) |
| même **`norm_stats.json`** | il fait partie du modèle : sha256 du fichier à côté du checkpoint |
| même entrée | même fichier CSV, ou même instant (`stem`) du même cache |
| même **`keep`** et même **graine** | `seed=0` par défaut ; l'API et la CLI n'exposent pas la graine, elles utilisent donc toujours 0 |
| même `tta` / `tta_keep` | 1 passe par défaut (`inference.tta`) |

**Tolérance retenue : 0 (égalité bit à bit).** Sur CPU, aucune étape du calcul n'est
aléatoire : le masquage des capteurs et les passes TTA tirent leurs capteurs avec
`np.random.default_rng(seed)`, et les noyaux CPU de torch (FFT, convolutions) sont
déterministes à entrée égale. Il n'y avait donc aucune raison de relâcher la tolérance :
une différence, même de 1e-7 m/s, signalerait un changement réel (de version, de poids,
de statistiques ou d'entrée).

Mesure complémentaire (mini-jeu, torch 2.2.2+cpu) : le résultat est aussi identique bit à
bit avec 1, 2 ou 4 threads. Ce n'est **pas** une garantie : sur un modèle plus gros ou une
autre bibliothèque BLAS / MKL, le découpage des réductions entre threads peut changer
l'ordre des additions flottantes. Pour une comparaison stricte, fixer le nombre de threads
(`torch.set_num_threads`, `OMP_NUM_THREADS`).

Portée : l'égalité bit à bit est **vérifiée sur le mini-modèle** du jeu de test. Pour le modèle v7
complet sur CPU, elle est attendue mais doit être rejouée avec la procédure « Rejouer » ci-dessous.

Reprise d'un entraînement sélectionné sur les DSO (`select_by: dso`) : le meilleur score enregistré
dans un checkpoint antérieur à la règle DSO3 (0,9 ; 1,1) a été calculé avec l'ancienne formule,
donc un peu plus bas ; les nouvelles époques lui sont comparées, et la sélection est légèrement plus
stricte. Sans effet sur v7, qui n'est pas réentraîné.

## Le test qui le vérifie

`tests/test_reproductibilite.py` (CPU, mini-jeu de `tests/conftest.py`), constante
`TOLERANCE_M_S = 0.0` :

- deux `WindFieldModel` chargés séparément depuis le même checkpoint donnent le même volume
  (`keep=0.5`, même graine), et le sha256 du checkpoint ne change pas ;
- le même modèle appelé deux fois donne le même volume (`keep` 1,0 et 0,5) ;
- la moyenne de passes (`tta=3`) est identique à graine égale, différente à graine différente ;
- le masquage des capteurs ne dépend que de la graine (graine par défaut = 0) ;
- le jeu d'évaluation (`ANEO3D`, `train=False`) masque les mêmes capteurs à chaque lecture.

```bash
python -m pytest -q tests/test_reproductibilite.py
```

## Sources de non-déterminisme connues (hors garantie)

- **GPU** : cuDNN choisit ses algorithmes à l'exécution, certaines réductions CUDA utilisent
  des additions atomiques, et les produits matriciels peuvent passer en **TF32**. Sur CUDA,
  le paquet calcule en plus en **bf16** (`autocast`) hors des deux îlots fp32 (FFT et tête
  de projection). Deux exécutions GPU peuvent donc différer à quelques 1e-3 m/s près, et un
  résultat GPU n'est pas comparable bit à bit à un résultat CPU.
- **Versions** de torch, numpy, MKL / OpenBLAS : une mise à jour peut changer l'ordre des
  opérations flottantes. « Même version » s'entend au sens de ces bibliothèques aussi.
- **Ordre des threads** : voir ci-dessus ; non observé sur le mini-jeu.
- **Entraînement** : il n'est **pas** reproductible bit à bit, et ce n'est pas l'objet de DE2.
  `train.seed` fixe `random`, `numpy` et `torch`, mais en mode entraînement `ANEO3D` tire la
  fraction de capteurs et le masque avec `np.random.default_rng()` **sans graine** (exprès :
  chaque exemple voit un réseau différent), et l'entraînement tourne sur GPU en bf16. Deux
  entraînements donnent des modèles équivalents, pas identiques. La reproductibilité porte
  donc sur un checkpoint **donné**, identifié par son sha256.

## Preuve historique

Sur Colab (A100, bf16, 2026-10-12), étape 0 du notebook v8
(`notebooks/current/pcfno3d_v8_colab.ipynb`) : le paquet a réévalué `best_dso_v7.pt` sur les
**120 instants synoptiques** de test et retrouvé la table du BLOC 10 du notebook v7 aux
**12 niveaux avec un écart maximal de 0,0000 m/s** sur les RMSE u et v par niveau (u 0,665 /
v 0,508 à 1000 m, global u 0,542) — voir `tukhnanutha.md` §5, « v8 notebook, first run ».
Le seuil d'arrêt fixé à l'avance était 0,01 m/s. C'est une preuve sur des métriques
agrégées, sur GPU : elle montre que le paquet reproduit le notebook, la garantie bit à bit
ci-dessus portant, elle, sur le CPU.

## Rejouer

```bash
# 1. identifier ce que l'on compare
python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <run_dir>/best_dso_v7.pt
python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <run_dir>/norm_stats.json

# 2. deux reconstructions du même instant, puis comparaison des .npz
python -m pcfno3d -c local_v7 reconstruct --stem 017_120000 --device cpu
# copier le <nom>_3d.npz produit, relancer, puis comparer u, v, w des deux fichiers avec
# numpy.array_equal(a, b, equal_nan=True) : attendu True

# 3. le test automatique, sur le mini-jeu
python -m pytest -q tests/test_reproductibilite.py
```

Une évaluation complète (`pcfno3d evaluate`) est elle aussi déterministe à configuration
égale : les instants de test, les masques (graines 2000 pour les instants synoptiques,
2500 pour les épisodes, 3000 pour la robustesse) et les passes TTA (graine = numéro de
l'instant) sont fixés ; seul le temps d'inférence (DSO4) varie d'une exécution à l'autre.
Elle écrit dans `run_dir/eval_pcfno3d/<checkpoint>/`, sans jamais écraser les résultats du
notebook.
