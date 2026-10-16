"""Lot M8 — E5, empreinte carbone (cahier v9, section 19.1, p. 42 ; tache 33, p. 45).

« Energie et CO2 de l'entrainement et d'une reconstruction (CodeCarbon), compares a une
simulation numerique. »

Trois sources, de la plus directe a la moins directe ; chaque chiffre porte sa `source` :

    codecarbon      si le paquet est installe : energie (CPU + GPU + RAM) et CO2 mesures
                    par CodeCarbon (`EmissionsTracker`)
    nvidia-smi      sinon, si `nvidia-smi` repond : puissance du GPU echantillonnee
                    (power.draw) et integree dans le temps -> energie du GPU SEUL, mesuree
    puissance_declaree
                    sinon, si l'utilisateur donne `--gpu-watts` : temps x puissance, une
                    ESTIMATION (majorant si la puissance est la puissance nominale)

Sans aucune des trois, seul le temps est rapporte et l'energie vaut None, marquee « a
mesurer sur GPU ». Le CO2 n'est calcule hors CodeCarbon que si l'intensite carbone
(gCO2/kWh) est fournie : aucun facteur n'est suppose ici.

L'entrainement de la v7 est deja fait : on ne peut en reconstruire honnetement que la
DUREE, a partir des journaux (sortie du BLOC 9 de `fno3d_v7.ipynb`, ou `history.json` du
paquet). Les epoques non journalisees sont comptees comme telles, jamais inventees.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path


def codecarbon_disponible() -> bool:
    import importlib.util

    return importlib.util.find_spec("codecarbon") is not None


def nvidia_smi_disponible() -> bool:
    return shutil.which("nvidia-smi") is not None


def _puissance_gpu_w() -> float | None:
    """Somme de power.draw (W) de tous les GPU visibles, None si illisible."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=5)
        vals = [float(x) for x in out.stdout.split() if x.strip()]
        return sum(vals) if vals else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class _Echantillonneur(threading.Thread):
    """Lit la puissance du GPU toutes les `pas_s` secondes (thread de fond)."""

    def __init__(self, pas_s: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.pas_s = pas_s
        self.mesures: list[tuple[float, float]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            p = _puissance_gpu_w()
            if p is not None:
                self.mesures.append((time.perf_counter(), p))
            self._stop.wait(self.pas_s)

    def arreter(self) -> None:
        self._stop.set()
        self.join(timeout=10)

    def energie_kwh(self) -> float | None:
        """Integrale trapezes de la puissance ; None avec moins de 2 mesures."""
        m = self.mesures
        if len(m) < 2:
            return None
        j = sum(0.5 * (m[i][1] + m[i + 1][1]) * (m[i + 1][0] - m[i][0])
                for i in range(len(m) - 1))
        return j / 3.6e6


class Sobriete:
    """Mesure d'une phase (contexte `with`) : duree, puis energie et CO2 selon la meilleure
    source disponible (voir l'en-tete du module).

    utiliser_codecarbon : None = s'il est installe ; False = jamais (tests, CPU).
    """

    def __init__(self, phase: str, dossier: str | Path | None = None,
                 gpu_watts: float | None = None, intensite_gco2_kwh: float | None = None,
                 utiliser_codecarbon: bool | None = None, utiliser_nvidia_smi: bool | None = None,
                 pas_s: float = 1.0) -> None:
        self.phase = phase
        self.dossier = Path(dossier) if dossier else None
        self.gpu_watts = gpu_watts
        self.intensite = intensite_gco2_kwh
        self.cc = codecarbon_disponible() if utiliser_codecarbon is None else utiliser_codecarbon
        self.smi = (nvidia_smi_disponible() if utiliser_nvidia_smi is None
                    else utiliser_nvidia_smi)
        self.pas_s = pas_s
        self.resultat: dict | None = None

    def __enter__(self) -> "Sobriete":
        self._tracker = self._ech = None
        if self.cc:
            from codecarbon import EmissionsTracker
            kw = {"project_name": f"pcfno3d_{self.phase}", "log_level": "error",
                  "save_to_file": self.dossier is not None}
            if self.dossier is not None:
                self.dossier.mkdir(parents=True, exist_ok=True)
                kw["output_dir"] = str(self.dossier)
            self._tracker = EmissionsTracker(**kw)
            self._tracker.start()
        elif self.smi:
            self._ech = _Echantillonneur(self.pas_s)
            self._ech.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        duree = time.perf_counter() - self._t0
        r: dict = {"phase": self.phase, "duree_s": round(duree, 3), "energie_kwh": None,
                   "co2_kg": None, "source_energie": None, "source_co2": None,
                   "perimetre": None}
        if self._tracker is not None:
            co2 = self._tracker.stop()
            data = getattr(self._tracker, "final_emissions_data", None)
            r.update({"energie_kwh": getattr(data, "energy_consumed", None),
                      "co2_kg": float(co2) if co2 is not None else None,
                      "source_energie": "codecarbon", "source_co2": "codecarbon",
                      "perimetre": "CPU + GPU + RAM (CodeCarbon)"})
        elif self._ech is not None:
            self._ech.arreter()
            e = self._ech.energie_kwh()
            r.update({"energie_kwh": e, "source_energie": "nvidia-smi" if e is not None else None,
                      "perimetre": "GPU seul (power.draw echantillonne)",
                      "echantillons": len(self._ech.mesures)})
        if r["energie_kwh"] is None and self.gpu_watts:
            r.update({"energie_kwh": duree * float(self.gpu_watts) / 3.6e6,
                      "source_energie": "puissance_declaree",
                      "perimetre": f"GPU seul, estimation temps x {self.gpu_watts:g} W "
                                   "(puissance donnee par l'utilisateur)"})
        if r["co2_kg"] is None and r["energie_kwh"] is not None and self.intensite:
            r.update({"co2_kg": r["energie_kwh"] * float(self.intensite) / 1000.0,
                      "source_co2": f"energie x {self.intensite:g} gCO2/kWh (fourni)"})
        if r["energie_kwh"] is None:
            r["note"] = "energie a mesurer sur GPU (CodeCarbon ou nvidia-smi indisponibles)"
        elif r["co2_kg"] is None:
            r["note"] = "CO2 non calcule : intensite carbone non fournie (--intensite-gco2-kwh)"
        self.resultat = r


def par_unite(r: dict, n: int, unite: str = "reconstruction") -> dict:
    """Ramene une mesure de phase a une unite (n repetitions)."""
    n = max(int(n), 1)

    def div(x):
        return None if x is None else x / n

    return {"unite": unite, "n": n, "duree_s": r["duree_s"] / n,
            "energie_kwh": div(r["energie_kwh"]), "co2_kg": div(r["co2_kg"]),
            "source_energie": r["source_energie"], "source_co2": r["source_co2"],
            "perimetre": r["perimetre"]}


# ---------------------------------------------------------------------- entrainement
_EPOQUE = re.compile(r"Ep\s+(\d+)\s*/\s*(\d+)\s*\|.*\|\s*([\d.]+)\s*s\b")
_GPU = re.compile(r"GPU\s*:\s*(.+)$")
_REPRISE = re.compile(r"Reprise a l'epoque\s+(\d+)")


def _sorties_notebook(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for c in nb.get("cells", []):
        for o in c.get("outputs", []) if c.get("cell_type") == "code" else []:
            t = o.get("text")
            if t:
                out.append("".join(t) if isinstance(t, list) else str(t))
    return "\n".join(out)


def historique_entrainement(path: str | Path) -> dict:
    """Duree d'entrainement reconstruite depuis un journal : notebook .ipynb (lignes
    « Ep  61/140 | ... | 120 s ») ou history.json du paquet (cle time_s par epoque).

    Rapporte ce qui est JOURNALISE ; les epoques absentes du journal sont listees comme
    non journalisees, sans duree inventee."""
    path = Path(path)
    if path.suffix == ".ipynb":
        texte = _sorties_notebook(path)
        ep = {}
        total = None
        for m in _EPOQUE.finditer(texte):
            ep[int(m.group(1))] = float(m.group(3))
            total = int(m.group(2))
        gpu = next((m.group(1).strip() for m in map(_GPU.search, texte.splitlines()) if m),
                   None)
        reprise = _REPRISE.search(texte)
        source = "sortie du notebook (lignes d'epoque)"
    else:
        hist = json.loads(path.read_text(encoding="utf-8"))
        ep = {int(r["epoch"]) + 1: float(r["time_s"]) for r in hist
              if isinstance(r, dict) and r.get("epoch", -1) >= 0 and "time_s" in r}
        total = max(ep) if ep else None
        gpu, reprise = None, None
        source = "history.json du paquet (time_s par epoque)"
    durees = list(ep.values())
    manquantes = sorted(set(range(1, (total or 0) + 1)) - set(ep)) if total else []
    return {"fichier": str(path.name), "source": source, "gpu": gpu,
            "epoques_prevues": total, "epoques_journalisees": len(ep),
            "premiere_journalisee": min(ep) if ep else None,
            "derniere_journalisee": max(ep) if ep else None,
            "reprise_a_l_epoque": int(reprise.group(1)) if reprise else None,
            "epoques_non_journalisees": len(manquantes),
            "duree_journalisee_s": round(sum(durees), 1) if durees else None,
            "duree_moyenne_epoque_s": round(sum(durees) / len(durees), 1) if durees else None,
            "note": "duree des epoques journalisees seulement ; les epoques non journalisees "
                    "(session precedente) ne sont pas estimees"}


def energie_entrainement(hist: dict, gpu_watts: float | None = None,
                         intensite_gco2_kwh: float | None = None) -> dict:
    """Energie de l'entrainement : seulement une ESTIMATION temps x puissance declaree,
    sur les epoques journalisees ; None sans puissance (a mesurer)."""
    t = hist.get("duree_journalisee_s")
    if t is None or not gpu_watts:
        return {"energie_kwh": None, "co2_kg": None,
                "note": "a mesurer : la puissance du GPU pendant l'entrainement v7 n'a pas "
                        "ete journalisee ; donner --gpu-watts pour une estimation"}
    e = t * float(gpu_watts) / 3.6e6
    return {"energie_kwh": e,
            "co2_kg": e * float(intensite_gco2_kwh) / 1000.0 if intensite_gco2_kwh else None,
            "source_energie": "puissance_declaree",
            "perimetre": f"GPU seul, {hist['epoques_journalisees']} epoques journalisees x "
                         f"{gpu_watts:g} W (estimation)"}
