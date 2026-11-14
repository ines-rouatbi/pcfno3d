"""Service ATOM Wind local, au format de production, sans Colab, sans poids reels et sans reseau.

    python scripts/dev_service.py                 # construit (une fois) puis sert sur http://127.0.0.1:8000
    python scripts/dev_service.py --port 8010 --refaire
    PCFNO3D_MOT_DE_PASSE_ADMIN=... python scripts/dev_service.py   # + un compte administrateur local

But : developper et tester l'interface (webapp/atom-wind) contre les VRAIES routes /api/atom-wind/*,
avec des reponses de la meme forme qu'en production, quand le service Colab dort. Ce n'est pas un
modele : le jeu est synthetique, le reseau un petit PC-FNO3D entraine quelques pas sur CPU, et les
sources temps reel (GFS Open-Meteo, METAR, SYNOP, GEFS) sont simulees localement. Les valeurs n'ont
aucune valeur meteorologique ; seuls les formats, les routes et les etats du service sont reels.

Etapes (chacune sautee si son resultat existe, --refaire recommence tout) :
  1. jeu synthetique sur la zone du Grand Tunis (68 x 56 km, noeuds a 1 km) : make_mini_dataset.py ;
  2. cache des instants (pcfno3d build-cache) ;
  3. forcage ERA5 synthetique (forcage_era5.csv/.npz) sur des cellules 0,25 deg couvrant la zone ;
  4. mode production « era5_stations » : 3 stations aux vraies positions (Tunis-Carthage, Zaghouan,
     Nabeul), vent ERA5 10 m (_E), sortie residuelle ;
  5. entrainement court (config `test`) ;
  6. service : uvicorn en processus, mode ouvert sur la boucle locale seulement, sources simulees.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

# origine du domaine reel (coin sud-ouest du Grand Tunis, cf. configs/grand_tunis.yaml)
LAT0, LON0 = 36.395, 10.04
LAT_E = np.array([36.25, 36.5, 36.75, 37.0])
LON_E = np.array([9.75, 10.0, 10.25, 10.5, 10.75, 11.0])
HAUTEURS = np.array([5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 650, 800, 3000], float)
STATIONS = [("DTTA Tunis-Carthage", 36.8504, 10.2156), ("Zaghouan (INM 60729)", 36.4333, 10.0833),
            ("Nabeul (INM 60728)", 36.455, 10.715)]
NIVEAUX = {1000: 110.0, 975: 330.0, 950: 560.0, 925: 800.0, 900: 1030.0, 850: 1500.0,
           800: 1970.0, 750: 2460.0, 700: 3100.0}


# ---------------------------------------------------------------------- construction du jeu
def construire(dossier: Path, refaire: bool):
    from pcfno3d.cache import build_cache
    from pcfno3d.config import Config
    from pcfno3d.forcage import SCALAIRES
    from pcfno3d.geometry import Geometry
    from pcfno3d.realdata import GeoRef

    if refaire and dossier.exists():
        shutil.rmtree(dossier)
    data, cache, run = dossier / "data", dossier / "cache", dossier / "run"
    if not (data / "meta.json").exists():
        print("1/5 jeu synthetique sur la zone du Grand Tunis...")
        subprocess.run([sys.executable, str(REPO / "scripts" / "make_mini_dataset.py"), str(data),
                        "--nx", "136", "--ny", "112", "--dx", "500", "--pente", "0.5"], check=True)

    cfg = Config.load("test")
    cfg.apply_overrides({"paths.data_dir": data, "paths.cache_dir": cache, "paths.run_dir": run,
                         "train.num_workers": 0})
    cfg.paths.mkdirs()
    geom = Geometry(cfg, device="cpu")
    print("2/5 cache des instants...")
    index = build_cache(cfg, geom, verbose=False)

    forcage = dossier / "forcage" / "forcage_era5.csv"
    if not forcage.exists():
        print("3/5 forcage ERA5 synthetique...")
        forcage.parent.mkdir(parents=True, exist_ok=True)
        t = pd.to_datetime(index["timestamp"])
        temps = pd.date_range(t.min().floor("D") - pd.Timedelta(hours=3),
                              t.max().ceil("D") + pd.Timedelta(hours=3), freq="3h")
        heures = ((temps - temps[0]) / pd.Timedelta(hours=1)).to_numpy()
        f = pd.DataFrame({"timestamp_utc": temps.strftime("%Y-%m-%dT%H:%M:%SZ")})
        for c, _ in SCALAIRES:
            f[c] = 0.5 * np.sin(heures / 24 * 2 * math.pi)
        f.to_csv(forcage, index=False)
        h, la, lo = np.meshgrid(HAUTEURS, LAT_E, LON_E, indexing="ij")
        u = 3.0 + 0.004 * h + 4.0 * (lat_rel := la - LAT0)
        v = -1.0 + 0.002 * h + 3.0 * (lo - LON0)
        del lat_rel
        n = len(temps)
        np.savez(forcage.with_suffix(".npz"), temps=temps.as_unit("s").astype("int64").to_numpy(),
                 lat=LAT_E, lon=LON_E, hauteurs_sol=HAUTEURS,
                 u=np.broadcast_to(u, (n,) + u.shape).astype(np.float32),
                 v=np.broadcast_to(v, (n,) + v.shape).astype(np.float32))

    cfg.apply_overrides({"paths.forcage_era5": forcage, "real.origin_lat": LAT0, "real.origin_lon": LON0,
                         "mask.entree": "era5_stations", "mask.pas_era5_m": 3 * geom.dx,
                         "model.residuel_era5": True})
    g = Geometry(cfg, device="cpu")
    if not (data / "stations.json").exists():
        print("4/5 mode production : stations, vent ERA5 10 m...")
        ref = GeoRef.from_cfg(cfg)
        jj, ii = np.nonzero(g.valid.any(axis=0))
        out = []
        for nom, la, lo in STATIONS:
            x, y = ref.to_xy(la, lo)
            k = int(np.argmin((ii * g.dx - x) ** 2 + (jj * g.dx - y) ** 2))
            out.append({"nom": nom, "iy": int(jj[k]), "ix": int(ii[k])})
        (data / "stations.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        for stem in index["stem"]:
            S = np.load(cache / f"{stem}_S.npy")
            np.save(cache / f"{stem}_E.npy", S[:2].astype(np.float16))

    if not (run / "best.pt").exists():
        print("5/5 entrainement court sur CPU...")
        from pcfno3d.train import Trainer
        Trainer(cfg, device="cpu", verbose=False).fit()
    # administrateur local (tests navigateur, essais) : seulement sur demande, mot de passe par l'environnement
    mdp = os.environ.get("PCFNO3D_MOT_DE_PASSE_ADMIN")
    if mdp:
        from service.auth import ApiKeyStore, default_store_path
        from service.comptes import Comptes
        email = os.environ.get("PCFNO3D_ADMIN_EMAIL", "admin@atomwind.local")
        Comptes(run, ApiKeyStore(default_store_path(run))).creer_admin(email, mdp)
        print(f"administrateur local : {email}")
    return cfg


# ---------------------------------------------------------------------- sources simulees
def _vitesse(t: pd.Timestamp, i: int) -> tuple[float, float]:
    """Vent 10 m simule : cycle diurne (brise l'apres-midi) et un peu d'ecart entre les points."""
    h = t.hour + t.minute / 60
    s = 2.5 + 4.0 * max(0.0, math.sin(math.pi * (h - 8) / 12)) + 0.35 * (i % 5)
    d = (250.0 + 70.0 * math.sin(math.pi * (h - 6) / 12)) % 360
    return s, d


def source_simulee(url: str) -> bytes:
    u = urllib.parse.urlparse(url)
    q = dict(urllib.parse.parse_qsl(u.query))
    if u.netloc == "api.open-meteo.com":
        temps = pd.date_range(q["start_date"], pd.Timestamp(q["end_date"]) + pd.Timedelta(hours=23), freq="h")
        pts = []
        for i, _ in enumerate(q["latitude"].split(",")):
            sv = [_vitesse(t, i) for t in temps]
            h = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in temps],
                 "wind_speed_10m": [round(s, 2) for s, _ in sv],          # m/s (wind_speed_unit=ms)
                 "wind_direction_10m": [d for _, d in sv],
                 "wind_gusts_10m": [round(s * 1.45, 1) for s, _ in sv],
                 "wind_speed_100m": [round(s * 1.35, 1) for s, _ in sv],
                 "wind_direction_100m": [(d + 8) % 360 for _, d in sv],
                 "temperature_2m": [round(17 + 7 * math.sin(math.pi * (t.hour - 8) / 14) + (i % 2), 1) for t in temps],
                 "relative_humidity_2m": [round(70 - 25 * math.sin(math.pi * (t.hour - 8) / 14)) for t in temps],
                 "surface_pressure": [1010.0] * len(temps), "pressure_msl": [1015.0] * len(temps),
                 "shortwave_radiation": [max(0.0, round(750 * math.sin(math.pi * (t.hour - 6) / 13), 1)) for t in temps]}
            for p, z in NIVEAUX.items():
                k = (1000 - p) / 300
                h[f"wind_speed_{p}hPa"] = [round(s * (1.6 + k), 1) for s, _ in sv]
                h[f"wind_direction_{p}hPa"] = [(d + 20 * k) % 360 for _, d in sv]
                h[f"geopotential_height_{p}hPa"] = [z] * len(temps)
            pts.append({"elevation": 0.0 if i % 3 == 0 else 60.0 + 40 * (i % 4), "hourly": h})
        return json.dumps(pts).encode()
    if "ensemble-api" in u.netloc:
        temps = pd.date_range(q["start_date"], pd.Timestamp(q["end_date"]) + pd.Timedelta(hours=23), freq="h")
        pts = []
        for i, _ in enumerate(q["latitude"].split(",")):
            h = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in temps]}
            for m in range(31):
                suf = "" if m == 0 else f"_member{m:02d}"
                h[f"wind_speed_10m{suf}"] = [round(_vitesse(t, i)[0] * (0.7 + 0.02 * m), 1) for t in temps]
                h[f"wind_direction_10m{suf}"] = [_vitesse(t, i)[1] for t in temps]
            pts.append({"hourly": h})
        return json.dumps(pts).encode()
    if "aviationweather" in u.netloc:
        maintenant = pd.Timestamp.now(tz="UTC").floor("h")
        rap = []
        for k in range(int(q.get("hours", 30))):
            t = maintenant - pd.Timedelta(hours=k)
            s, d = _vitesse(t + pd.Timedelta(hours=1), 0)
            rap.append({"icaoId": "DTTA", "reportTime": t.isoformat(), "wspd": round(s * 1.944), "wdir": round(d / 10) * 10})
        return json.dumps(rap).encode()
    if "ogimet" in u.netloc:
        maintenant = pd.Timestamp.now(tz="UTC").floor("h")
        lignes = []
        for k in range(0, 30, 3):
            t = maintenant - pd.Timedelta(hours=k)
            for code, i in (("60729", 1), ("60728", 2)):
                s, d = _vitesse(t + pd.Timedelta(hours=1), i)
                lignes.append(f"{code},{t:%Y},{t:%m},{t:%d},{t:%H},00,AAXX {t:%d}{t:%H}4 {code} 16/// "
                              f"/{round(d / 10) % 36:02d}{min(round(s * 1.944), 98):02d} 10174 20112=")
        return ("\n".join(lignes) + "\n").encode()
    raise ConnectionError(f"service local : source non simulee ({u.netloc})")


# ---------------------------------------------------------------------- service
def servir(cfg, port: int) -> None:
    import uvicorn

    from pcfno3d import temps_reel
    import service.app as appmod

    temps_reel._get = lambda url, timeout=30.0: source_simulee(url)
    appmod._load_config = lambda: cfg
    os.environ.pop("PCFNO3D_INSECURE_OPEN", None)
    os.environ["PCFNO3D_PORT"] = str(port)
    os.environ.setdefault("PCFNO3D_DEVICE", "cpu")
    print(f"\nATOM Wind (service local, donnees simulees) : http://127.0.0.1:{port}/app/atom-wind/")
    uvicorn.run(comme_le_worker(appmod.create_app(), cle_visiteur(cfg)), host="127.0.0.1", port=port,
                log_level="warning")


def cle_visiteur(cfg) -> str:
    """La cle de lecture que le Worker d'atom.zelqane.com injecte pour les visiteurs sans compte ; gardee dans
    le dossier local du service (artifacts/, hors git), creee au premier lancement."""
    from service.auth import ApiKeyStore, default_store_path
    f = Path(cfg.paths.run_dir) / ".cle_visiteur"
    store = ApiKeyStore(default_store_path(cfg.paths.run_dir))
    if f.exists() and store.verify(f.read_text(encoding="utf-8").strip()) is not None:
        return f.read_text(encoding="utf-8").strip()
    cle, _ = store.create(f"worker-local-{os.getpid()}", scope="read")
    f.write_text(cle, encoding="utf-8")
    return cle


def comme_le_worker(app, cle: str):
    """Comme le Worker de production : une requete /api sans cle recoit la cle de lecture ; une session de
    compte (X-ATOM-Session) devient la cle presentee. Le service lui-meme reste ferme par defaut."""
    async def asgi(scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/api"):
            h = [(k, v) for k, v in scope["headers"] if k not in (b"x-api-key",)]
            sess = next((v for k, v in scope["headers"] if k == b"x-atom-session"), None)
            pres = next((v for k, v in scope["headers"] if k == b"x-api-key"), None)
            h.append((b"x-api-key", sess or pres or cle.encode()))
            scope = {**scope, "headers": h}
        await app(scope, receive, send)
    return asgi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dossier", type=Path, default=REPO / "artifacts" / "dev")
    ap.add_argument("--refaire", action="store_true", help="tout reconstruire")
    ap.add_argument("--construire-seulement", action="store_true")
    a = ap.parse_args()
    cfg = construire(a.dossier.resolve(), a.refaire)
    if not a.construire_seulement:
        servir(cfg, a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
