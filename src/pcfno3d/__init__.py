"""PC-FNO3D — reconstruction du champ de vent 3D sur la Tunisie a partir du reseau ANEO.

Portage du notebook `notebooks/current/fno3d_v7_1.ipynb` (BLOC 0 a BLOC 12) en paquet
importable, pour pouvoir entrainer, evaluer, reconstruire et servir le modele hors Colab.

Correspondance notebook -> module :
    BLOC 0/1   -> pcfno3d.config      (Config, chemins, hyperparametres, seuils DSO)
    BLOC 2     -> pcfno3d.geometry    (grille, masques, operateurs discrets)
    BLOC 3/4   -> pcfno3d.cache       (CSV journaliers -> tenseurs .npy + index)
    BLOC 5     -> pcfno3d.normalize   (statistiques de normalisation)
    BLOC 6     -> pcfno3d.features / pcfno3d.dataset
    BLOC 7     -> pcfno3d.model
    BLOC 8     -> pcfno3d.losses
    BLOC 9/9b  -> pcfno3d.train
    BLOC 10    -> pcfno3d.evaluate
    BLOC 11/12 -> pcfno3d.reconstruct (+ pcfno3d.export pour le web)
"""

__version__ = "0.3.2"

__all__ = ["__version__"]
