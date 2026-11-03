# Rapport de PFE — PC-FNO3D / ATOM Wind (Ines Rouatbi)

Projet LaTeX prêt pour Overleaf.

## Importer dans Overleaf

1. Se connecter à Overleaf avec le compte ines.rouatbi14@gmail.com.
2. **New Project → Upload Project**, choisir `rapport_pfe_atom_wind.zip`.
3. Menu → Settings : Compiler = **pdfLaTeX**, Main document = `main.tex` (valeurs par défaut).
4. Recompiler : 49 pages environ.

## À compléter (en rouge dans le PDF, commande `\acompleter{...}`)

- page de garde : spécialité / option, encadrants académique et professionnel ;
- logos (ESPRIT, Tukhnanutha) : déposer les images dans `figures/` et décommenter les lignes
  `\includegraphics` de `chapitres/00_pages_liminaires.tex` ;
- dédicace, remerciements, apports personnels (conclusion) ;
- présentation de Tukhnanutha (date de création, effectif, activités) ;
- organisation Scrum (section 1.4.3) : outil de suivi utilisé et cadence réelle des revues ;
- captures d'écran de l'application ATOM Wind (chapitre 6) ;
- résultats d'OS7 (PREV1–3, AGRI1–2), à reporter quand le calcul Colab est terminé.

## Structure

- `main.tex` : préambule et ordre des chapitres ;
- `chapitres/` : pages liminaires, introduction, chapitres 1 à 6, conclusion ;
- `annexes/` : configuration du modèle livré et commandes ;
- `figures/` : figures (tracées par `outils/figures.py` à partir des résultats archivés) ;
- `references.bib` : bibliographie (BibTeX, style `plainnat`).

Tous les chiffres viennent des résultats archivés du projet (fiches modèle, bilan final,
`decision_v4.json`, `os6.json`, bilan de conformité). Le cahier des charges est confidentiel :
le rapport le paraphrase sans le reproduire.
