# Contrat v1 — copie de référence (ne pas modifier ici)

`schemas/` et `fixtures/` sont la copie **à l'identique, octet pour octet** de `contract/schemas/`
et `contract/fixtures/` du dépôt `rayenbenamorr/atom-dashboard`, PR #1 (branche `pr1`),
commit source `61a0087166ac54ec61ce33ec8677c3ddc5862209`.

La copie a été extraite des objets git, sans conversion
(`git -c core.autocrlf=false archive HEAD contract/schemas contract/fixtures`), puis vérifiée
blob par blob (`git hash-object --no-filters`) : sous Windows, `core.autocrlf` convertit les
JSON en CRLF — y compris dans `git archive` — et le sha256 inscrit dans `publish-manifest.json`
ne correspond plus. `.gitattributes` marque ce dossier `-text` pour la même raison.

**Toute modification passe par une PR des deux côtés** : d'abord dans `atom-dashboard/contract/`
(relue par le service et par le tableau de bord), puis recopie ici avec le nouveau hash de commit.
Ne jamais corriger un schéma ou une fixture directement dans ce dossier.

Ce qui l'utilise : `tests/test_contract_v1.py`, qui valide les fixtures et, surtout, la sortie
actuelle du service convertie par `src/pcfno3d/contract_v1.py`. Les ressources que le service
ne sait pas encore produire y sont marquées `xfail(strict=True)` avec le lot qui les livrera
(M2 fiabilité, M3 contrat ANEO, M4 alertes, M5 éolien) : quand un lot les rend conformes, le
test passe, `strict` le fait échouer, et il faut retirer le marqueur — la barrière se resserre.
