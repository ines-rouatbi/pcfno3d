# Validation complete du pipeline sur le mini-dataset, sans Colab et sans GPU.
# Equivalent de ANEO_TEST=1 dans le notebook, mais de bout en bout.
#
#   powershell -ExecutionPolicy Bypass -File scripts\smoke.ps1
#
# Duree : moins d'une minute. Ne verifie pas la qualite du modele (2 epoques), mais que
# chaque etape s'execute et produit ses fichiers.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$env:PYTHONPATH = "src;."
$env:PYTHONIOENCODING = "utf-8"

function Step($titre) { Write-Host "`n=== $titre ===" -ForegroundColor Cyan }

Step "1/6 mini-dataset"
python scripts/make_mini_dataset.py artifacts/mini

Step "2/6 cache de tenseurs"
python -m pcfno3d -c test build-cache

Step "3/6 controle geometrie et verite terrain"
python -m pcfno3d -c test check

Step "4/6 entrainement (2 epoques, modele minuscule)"
python -m pcfno3d -c test train

Step "5/6 evaluation DSO"
python -m pcfno3d -c test evaluate

Step "6/6 reconstruction 3D + export web"
python -m pcfno3d -c test reconstruct --altitude 120 --web

Step "tests unitaires"
python -m pytest -q

Write-Host "`nPipeline valide. Artefacts dans artifacts\mini_run" -ForegroundColor Green
Write-Host "Visualiseur : python -m http.server 8080 puis" -ForegroundColor Green
Write-Host "  http://localhost:8080/web/?src=../artifacts/mini_run/web/recon_006_000000"
