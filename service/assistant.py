"""Assistant conversationnel integre (cahier v12, E6) : l'utilisateur d'ATOM Wind pose une question en
langage naturel (« peut-on traiter les vignes de Mornag demain matin ? ») ; Claude y repond en appelant
les outils du service (service.atom_outils : vent en un point, profil, decision agricole, production
eolienne, alertes), comme le serveur MCP, sans calcul de plus.

Cle : variable d'environnement ANTHROPIC_API_KEY du service (jamais dans le depot ni dans la page).
Modele : Claude Opus 5.5, effort « low » (questions courtes de conversation), repli automatique vers un
autre modele en cas de refus (`fallbacks: "default"`), cache du prompt systeme et des outils.
"""
from __future__ import annotations

import json
import os

MODELE = os.environ.get("PCFNO3D_ASSISTANT_MODELE", "claude-opus-5-5")
EFFORT = os.environ.get("PCFNO3D_ASSISTANT_EFFORT", "low")
MAX_TOURS_OUTILS = 8
BETA_REPLI = "server-side-fallback-2026-07-01"

SYSTEME = """Tu es l'assistant d'ATOM Wind, l'application de vent du Grand Tunis (zone de 68 x 56 km entre \
Tunis, Zaghouan et Nabeul) de Tukhnanutha. Le vent y est calcule par le modele PC-FNO3D : un champ 3D \
du sol a 1 000 m, a 100 m de resolution, a partir de la grande echelle GFS et des anemometres de \
Tunis-Carthage, Zaghouan et Nabeul. Aujourd'hui = temps reel ; demain et apres-demain = prevision.

Reponds en francais (ou dans la langue de la question), en phrases simples, pour des agriculteurs, des \
exploitants eoliens ou le grand public. Commence par la reponse a la question en une phrase, puis \
donne les chiffres utiles (vent en km/h au sol, en m/s pour l'eolien, heure locale de Tunis).

Utilise les outils pour toute valeur : n'invente jamais un chiffre. Si un lieu n'est pas reconnu, dis \
quels lieux tu connais ou demande ses coordonnees. Dis d'ou vient chaque grandeur : le vent vient \
d'ATOM Wind ; la temperature et l'humidite viennent de GFS. Pour l'eolien, rappelle que la production \
est une estimation pour une eolienne de reference, sans mesure de production. Pour un traitement \
phytosanitaire, applique la regle de l'application (oui de 3 a 15 km/h a 10 m) et rappelle de \
respecter l'etiquette du produit et les zones non traitees. Si l'outil signale une donnee absente \
ou le service hors ligne, dis-le simplement."""


def _nullable(t: str, desc: str) -> dict:
    return {"type": [t, "null"], "description": desc}


_LIEU = {"lieu": _nullable("string", "Nom d'un lieu de l'application (Mornag, Nabeul, Soliman, Tunis-Carthage...) ; null si lat/lon"),
         "lat": _nullable("number", "Latitude en degres (null si lieu)"),
         "lon": _nullable("number", "Longitude en degres (null si lieu)")}
_JOUR = {"jour": {"type": "integer", "enum": [0, 1, 2], "description": "0 aujourd'hui, 1 demain, 2 apres-demain"}}


def _outil(nom: str, description: str, props: dict) -> dict:
    return {"name": nom, "description": description, "strict": True,
            "input_schema": {"type": "object", "properties": props, "required": list(props),
                             "additionalProperties": False}}


OUTILS = [
    _outil("vent_au_point", "Vent (vitesse, direction, temperature et humidite GFS) en un point, a une hauteur "
           "au-dessus du sol et une heure locale, aujourd'hui, demain ou apres-demain.",
           {**_LIEU, **_JOUR, "heure": {"type": "number", "description": "Heure locale de Tunis, 0 a 23"},
            "hauteur_m": {"type": "number", "description": "Hauteur au-dessus du sol en metres (2 a 1000)"}}),
    _outil("profil_vertical", "Vent a toutes les hauteurs (10 a 1000 m au-dessus du sol) en un point et une heure.",
           {**_LIEU, **_JOUR, "heure": {"type": "number", "description": "Heure locale de Tunis, 0 a 23"}}),
    _outil("decision_agricole", "Creneaux de traitement et d'aspersion d'une parcelle pour une journee, heure par "
           "heure, avec la probabilite de pouvoir traiter (ensemble GEFS) si elle est calculee.",
           {**_LIEU, **_JOUR}),
    _outil("production_eolienne", "Production heure par heure et energie du jour d'une eolienne de reference de "
           "2 MW placee en un point, a une hauteur de moyeu.",
           {**_LIEU, **_JOUR, "moyeu_m": {"type": "number", "description": "Hauteur de moyeu au-dessus du sol (30 a 150 m)"}}),
    _outil("alertes_vent", "Vent le plus fort de la journee sur tout le Grand Tunis, a 10 m et a 100 m, avec le lieu "
           "et l'heure, et les alertes de vent fort.", {**_JOUR}),
]
OUTILS[-1]["cache_control"] = {"type": "ephemeral"}       # le prefixe outils + systeme est stable


def executer(wf, nom: str, entree: dict) -> dict:
    from service import atom_outils as ao
    lieu = {k: entree.get(k) for k in ("lieu", "lat", "lon")}
    j = int(entree.get("jour", 0))
    if nom == "vent_au_point":
        return ao.lire_point(wf, j, float(entree["hauteur_m"]), float(entree["heure"]), **lieu)
    if nom == "profil_vertical":
        return ao.profil(wf, j, float(entree["heure"]), **lieu)
    if nom == "decision_agricole":
        d = ao.decision_agricole(wf, j, **lieu)
        d["heures"] = [h for h in d["heures"] if 5 <= h["heure"] <= 20]      # assez pour repondre
        return d
    if nom == "production_eolienne":
        return ao.production_eolienne(wf, j, float(entree["moyeu_m"]), **lieu)
    if nom == "alertes_vent":
        return ao.alertes_vent(wf, j)
    raise ValueError(f"outil inconnu : {nom}")


class AssistantIndisponible(RuntimeError):
    pass


def client():
    try:
        import anthropic
    except ImportError as exc:
        raise AssistantIndisponible("SDK anthropic absent (pip install anthropic)") from exc
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise AssistantIndisponible("assistant non configure : ANTHROPIC_API_KEY absente sur le service")
    return anthropic.Anthropic()


def repondre(wf, question: str, historique: list[dict] | None = None, cl=None) -> dict:
    """Une reponse de l'assistant ; `historique` : [{"role": "user"|"assistant", "contenu": str}]."""
    cl = cl or client()
    messages = [{"role": m["role"], "content": m["contenu"]} for m in (historique or [])
                if m.get("role") in ("user", "assistant") and m.get("contenu")]
    messages.append({"role": "user", "content": question})
    appels = []
    for _ in range(MAX_TOURS_OUTILS + 1):
        r = cl.beta.messages.create(
            model=MODELE, max_tokens=4000, system=SYSTEME, tools=OUTILS, messages=messages,
            output_config={"effort": EFFORT}, cache_control={"type": "ephemeral"},
            betas=[BETA_REPLI], fallbacks="default")
        if r.stop_reason == "refusal":
            return {"reponse": "Je ne peux pas repondre a cette demande.", "outils": appels, "modele": r.model,
                    "arret": "refus"}
        if r.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": r.content})
            continue
        if r.stop_reason != "tool_use":
            texte = "\n".join(b.text for b in r.content if b.type == "text").strip()
            if r.stop_reason == "max_tokens":
                texte += "\n\n(reponse coupee)"
            return {"reponse": texte, "outils": appels, "modele": r.model, "arret": r.stop_reason}
        messages.append({"role": "assistant", "content": r.content})
        resultats = []
        for b in r.content:
            if b.type != "tool_use":
                continue
            appels.append({"nom": b.name, "entree": b.input})
            try:
                res = {"type": "tool_result", "tool_use_id": b.id,
                       "content": json.dumps(executer(wf, b.name, dict(b.input)), ensure_ascii=False)}
            except Exception as exc:                                  # noqa: BLE001
                res = {"type": "tool_result", "tool_use_id": b.id, "content": f"Erreur : {exc}", "is_error": True}
            resultats.append(res)
        messages.append({"role": "user", "content": resultats})
    return {"reponse": "Je n'ai pas pu terminer la reponse (trop d'etapes).", "outils": appels, "modele": MODELE,
            "arret": "tours_max"}
