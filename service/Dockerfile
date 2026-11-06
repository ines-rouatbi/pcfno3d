# Image PC-FNO3D (CPU) : application web + API JSON + serveur MCP.
# Pour une image GPU, partir de pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime et retirer
# l'index CPU ci-dessous.
#
#   docker build -f service/Dockerfile -t pcfno3d-api .
#   docker run --rm -p 8000:8000 \
#     -v /chemin/vers/artifacts:/app/artifacts \
#     -v /chemin/vers/aneo_dataset_3d_v2:/data:ro \
#     -e PCFNO3D_DATA_DIR=/data \
#     -e PCFNO3D_API_KEYS=app-externe:pcf_... \
#     -e PCFNO3D_ALLOWED_HOSTS=mon-domaine.tn \
#     -v /chemin/vers/terrain:/terrain:ro \
#     pcfno3d-api
#
# Le relief 3D de la Carte 3D (~470 Mo, dossier contenant tunisia/) n'est pas dans l'image
# (.dockerignore) : il est monte sur /terrain et servi sous /app/terrain/. Sans lui, la
# Carte 3D l'indique et les autres vues fonctionnent ; /api/health dit terrain_3d.
#
# Le volume artifacts doit contenir runs/v7/{best_dso_v7.pt,norm_stats.json} et le cache
# des instants. PCFNO3D_ALLOWED_HOSTS est obligatoire derriere un proxy : le transport
# MCP refuse un en-tete Host non declare.
FROM node:24-slim AS frontend-build
WORKDIR /ui
COPY package.json package-lock.json vite.config.ts tsconfig.json ./
COPY frontend ./frontend
# webapp/terrain est exclu (.dockerignore) : le relief est un volume, pas une couche d'image.
COPY webapp ./webapp
COPY scripts/extract_frontend_markup.mjs ./scripts/extract_frontend_markup.mjs
COPY scripts/finalize_frontend.mjs ./scripts/finalize_frontend.mjs
COPY scripts/copy_frontend_vendor.mjs ./scripts/copy_frontend_vendor.mjs
RUN mkdir -p web
RUN npm ci && npm run build

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PCFNO3D_CONFIG=default \
    PCFNO3D_PORT=8000 \
    PCFNO3D_TERRAIN_DIR=/terrain

WORKDIR /app

# torch CPU : l'image reste bien plus petite que la variante CUDA
COPY requirements.txt ./requirements.txt
COPY service/requirements.txt ./service/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 \
 && pip install --no-cache-dir -r requirements.txt -r service/requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY service ./service
COPY configs ./configs
COPY --from=frontend-build /ui/webapp ./webapp
RUN pip install --no-cache-dir --no-deps -e .

# utilisateur non privilegie
RUN useradd --create-home --uid 10001 aneo && chown -R aneo /app
USER aneo

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
# application web : /app/   API : /api/docs   MCP : /mcp
