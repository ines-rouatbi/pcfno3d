"""Sources PlantUML des diagrammes du rapport corriges le 24/11/2026 (rendus par le serveur PlantUML en ligne :
python docs/uml/rapport/rendre.py)."""
from pathlib import Path

ICI = Path(__file__).resolve().parent
STYLE = """skinparam defaultFontName Helvetica
skinparam defaultFontSize 12
skinparam shadowing false
skinparam ArrowColor #2B6CB0
skinparam actorBorderColor #2B6CB0
skinparam actorBackgroundColor #FEFECE
skinparam usecaseBorderColor #2B6CB0
skinparam usecaseBackgroundColor #EAF2FD
skinparam rectangleBorderColor #2B4870
skinparam componentBorderColor #2B6CB0
skinparam componentBackgroundColor #EAF2FD
skinparam nodeBorderColor #4A5568
skinparam databaseBorderColor #2B4870
skinparam participantBorderColor #2B6CB0
skinparam participantBackgroundColor #EAF2FD
skinparam sequenceLifeLineBorderColor #2B6CB0
skinparam noteBackgroundColor #FFFBD7
"""

D = {}
D["cas_utilisation_global"] = r'''
left to right direction
actor "Visiteur" as V
actor "Agriculteur" as A
actor "Exploitant éolien" as E
actor "Administrateur" as AD
actor "Opérateur ANEO /\nData scientist" as O
actor "Assistant IA\n(client MCP)" as IA <<système>>
actor "Programme partenaire\n(clé d'API)" as P <<système>>
actor "Sources météo\n(GFS, METAR, SYNOP)" as S <<système>>
actor "Serveur de courriel\n(SMTP)" as M <<système>>
A --|> V
E --|> V
rectangle "ATOM Wind / service PC-FNO3D (Grand Tunis et Tunisie entière)" {
  usecase "Consulter la vue d'ensemble,\nles 24 gouvernorats et la vigilance" as UC1
  usecase "Consulter la carte et la vue 3D\ndu vent (J0 à J+2)" as UC2
  usecase "Lire le vent et le profil\nen un point" as UC3
  usecase "Installer l'application\n(ordinateur ou téléphone)" as UC4
  usecase "Créer un compte" as UC5
  usecase "Vérifier son adresse e-mail\n(lien de 24 h)" as UC6
  usecase "Se connecter\n(session de 8 h)" as UC7
  usecase "Mot de passe oublié\n(lien d'1 h)" as UC8
  usecase "Planifier traitement et irrigation\n(créneaux, besoin FAO-56)" as UC9
  usecase "Recevoir des alertes\n(appareil, courriel, SMS, WhatsApp)" as UC10
  usecase "Estimer la production\nd'un parc (sillage, atlas)" as UC11
  usecase "Gérer comptes, rôles\net clés d'API" as UC12
  usecase "Superviser le service\net la qualité du modèle" as UC13
  usecase "Planifier le réseau ANEO\n(mâts à 25 m)" as UC14
  usecase "Exporter (NetCDF, CSV)\net consulter l'historique" as UC15
  usecase "Entraîner et évaluer un modèle\n(règle hachée, test unique)" as UC16
  usecase "Décider avec « Mon vent »\n(activités propres au rôle)" as UC17
  usecase "Interroger l'API avec une clé\n(portée, quota, journal)" as UC18
  usecase "Reconstruire le champ 3D\nd'un instant" as R
}
V --> UC1
V --> UC2
V --> UC3
V --> UC4
V --> UC5
V --> UC7
V --> UC8
V --> UC17
A --> UC9
A --> UC10
E --> UC11
IA --> UC18
P --> UC18
UC18 ..> R : <<include>>
UC17 ..> R : <<include>>
note right of UC17
  visiteur et grand public : sortir et se déplacer
  agriculteur : traiter, arroser par aspersion
  exploitant éolien : produire avec une éolienne
  administrateur : toutes (contrôle)
end note
AD --> UC12
AD --> UC13
AD --> UC14
O --> UC14
O --> UC15
O --> UC16
UC5 ..> UC6 : <<include>>
UC6 ..> M
UC8 ..> M
UC9 ..> R : <<include>>
UC2 ..> R : <<include>>
UC3 ..> R : <<include>>
R --> S
'''

D["cas_utilisation_admin"] = r'''
left to right direction
actor "Administrateur" as AD
rectangle "Espace administration" {
  usecase "Valider une demande\nde rôle éolien" as C1
  usecase "Gérer les comptes : confirmer une adresse,\nrenvoyer le lien, changer le rôle" as C2
  usecase "Désactiver, réactiver\nou supprimer un compte" as C3
  usecase "Créer une clé d'API pour un programme\n(portée read ou admin, quota)" as C4
  usecase "Révoquer une clé et suivre\nson dernier usage" as C4b
  usecase "Consulter la supervision\n(stations, temps, résidu de masse)" as C5
  usecase "Consulter la qualité\ndu modèle servi (DSO)" as C6
  usecase "Gérer les abonnés SMS\net envoyer une notification" as C7
  usecase "Planifier le réseau ANEO\n(mâts à 25 m, plan en grille)" as C8
  usecase "Outils de terrain\n(grue, haies, mâts, panne de station)" as C9
  usecase "S'authentifier (rôle admin,\nrelu à chaque requête)" as AUTH
}
AD --> C1
AD --> C2
AD --> C3
AD --> C4
AD --> C4b
AD --> C5
AD --> C6
AD --> C7
AD --> C8
AD --> C9
C1 ..> AUTH : <<include>>
C2 ..> AUTH : <<include>>
C3 ..> AUTH : <<include>>
C4 ..> AUTH : <<include>>
C4b ..> AUTH : <<include>>
C7 ..> AUTH : <<include>>
C8 ..> AUTH : <<include>>
C9 ..> AUTH : <<include>>
'''

D["sequence_inscription"] = r'''
autonumber
actor "Utilisateur" as U
participant "Application\nATOM Wind" as APP
participant "Worker\nCloudflare" as W
participant "API FastAPI\n(Colab)" as API
participant "Comptes" as C
participant "Courriel\n(SMTP)" as M
actor "Administrateur" as AD
U -> APP : nom, e-mail, mot de passe, profil
APP -> W : POST /api/comptes
W -> API : relais
API -> C : creer(email, mdp, rôle)
C -> C : scrypt salé ; adresse NON vérifiée
C -> C : jeton aléatoire, 24 h\n(seule son empreinte SHA-256 est gardée)
API -> M : lien #verifier-email=jeton
API --> APP : 201 (aucune session ouverte)
APP --> U : « Vérifiez votre adresse e-mail »
alt connexion avant d'ouvrir le lien
  U -> API : connexion (bon mot de passe)
  API --> U : 403 adresse non vérifiée (renvoyer le lien)
end
M --> U : e-mail reçu (lien à usage unique)
U -> API : POST /api/comptes/verification (jeton)
API -> C : verifier_email(jeton)
C -> C : jeton valide, effacé
API --> APP : adresse confirmée : se connecter
opt rôle éolien demandé
  AD -> API : POST /api/comptes/{email}/validation
  API -> C : rôle eolien accordé
end
U -> API : connexion : session de 8 h, espaces du rôle
'''

D["sequence_connexion"] = r'''
autonumber
actor "Utilisateur" as U
participant "Application\nATOM Wind" as APP
participant "Worker\nCloudflare" as W
participant "API FastAPI\n(Colab)" as API
participant "Comptes" as C
participant "ApiKeyStore" as K
U -> APP : e-mail, mot de passe
APP -> W : POST /api/comptes/connexion
W -> API : relais + clé de lecture
API -> C : connecter(email, mot_de_passe)
alt 5 échecs en moins de 15 min
  C -> C : refus (compte bloqué 15 min)
  API --> APP : 429 Too Many Requests
else mot de passe faux ou compte désactivé
  C -> C : compter l'échec
  API --> APP : 401 Unauthorized
else adresse e-mail pas encore vérifiée
  API --> APP : 403 (ouvrir le lien reçu ou le renvoyer)
else succès
  C -> K : session compte-<sha12(email)>, expire dans 8 h
  K --> C : clé pcf_… (seule l'empreinte SHA-256 est stockée)
  C --> API : clé de session, compte
  API --> APP : 200 {cle, compte}
  APP -> APP : session gardée par onglet (en-tête X-ATOM-Session)
  APP --> U : espaces réservés au rôle
end
'''

D["c4_conteneurs"] = r'''
rectangle "Personnes" {
  actor "Visiteur" as V
  actor "Agriculteur" as A
  actor "Exploitant éolien" as E
  actor "Administrateur" as AD
}
rectangle "Système ATOM Wind (frontière)" {
  component "Application web installable (PWA)\n[React, TypeScript, Vite]\ncarte, 3D, gouvernorats, espaces métier,\ntrois langues, thème sombre" as APP
  component "Passerelles\n[Workers Cloudflare]\nclé de lecture injectée, dernier état" as W
  component "API REST\n[FastAPI]\nroutes, clés, comptes, rôles, quotas" as API
  component "Serveur MCP\n[MCP]\n23 outils" as MCP
  component "Moteur PC-FNO3D\n[PyTorch, paquet pcfno3d]\nGrand Tunis + Tunisie entière" as MOT
  database "Stockage\n[Google Drive, fichiers]\npoids, cache, historique,\ncomptes, clés (SHA-256)" as DB
}
rectangle "Systèmes externes" {
  component "Courriel (SMTP)\nliens de vérification\net de mot de passe" as SMTP
  component "Twilio\n(SMS, WhatsApp)" as TW
  component "NOAA GFS / Open-Meteo\n(GFS, GEFS)" as GFS
  component "METAR / SYNOP" as OBS
  component "Assistant IA\n(client MCP)" as IA
}
V --> APP
A --> APP
E --> APP
AD --> APP
APP --> W : HTTPS, JSON
W --> API : HTTPS (tunnel nommé)
API --> MOT
MCP --> MOT
API --> DB
MOT --> DB
API --> SMTP
API --> TW
API --> GFS
API --> OBS
IA --> MCP : HTTPS
'''

D["deploiement"] = r'''
node "Poste ou téléphone de l'utilisateur" {
  artifact "Navigateur ou application installée\n(ATOM Wind : manifeste, service worker)" as NAV
}
node "Poste de l'opérateur" {
  artifact "Client MCP / scripts\npcfno3d (CLI)" as CLI
}
cloud "Cloudflare" {
  node "Workers\natom.zelqane.com\natomwind.tukhnanutha.workers.dev" as WK {
    artifact "worker.js\n(clé injectée, cache du dernier état)" as WJ
  }
  node "Tunnel nommé\natom-origin.zelqane.com" as TUN
}
node "Google Colab (GPU A100 ou L4)" as COLAB {
  artifact "cloudflared" as CF
  artifact "uvicorn : service FastAPI\n+ serveur MCP (/mcp)" as UV
  artifact "WindFieldModel x 2\n(Grand Tunis, Tunisie entière)" as WF
}
database "Google Drive" as GD {
  artifact "poids, caches, historique,\ncomptes, journaux" as FIC
}
cloud "Services externes" {
  artifact "NOAA GFS / Open-Meteo" as OM
  artifact "aviationweather.gov (METAR)\nOGIMET (SYNOP)" as AW
  artifact "Serveur SMTP\n(courriels des comptes)" as SM
}
NAV --> WK : HTTPS
CLI --> WK : HTTPS (clé d'API)
WK --> TUN : HTTPS
TUN --> CF
CF --> UV : HTTP local
UV --> WF
WF --> GD : lecture
UV --> OM : HTTPS
UV --> AW : HTTPS
UV --> SM : SMTP (STARTTLS)
'''

D["sequence_cle_api"] = r'''
autonumber
actor "Administrateur" as AD
participant "Programme partenaire" as P
participant "Garde-fou\n(make_guard)" as G
database "Trousseau\n(empreintes SHA-256)" as K
participant "Route de l'API" as RT
database "Journal d'accès" as J
AD -> K : créer la clé (portée read, quota 5 / min)
K --> AD : secret pcf_… affiché une seule fois
AD -> P : transmet le secret (canal sûr)
P -> G : GET /api/atom-wind/domaines\nX-API-Key: pcf_…
G -> K : empreinte SHA-256 du secret connue et non révoquée ?
alt clé absente, fausse ou révoquée
  G --> P : 401
else quota de la minute dépassé
  G --> P : 429 + Retry-After
else clé valide
  G -> RT : requête (portée vérifiée par la route)
  alt portée insuffisante (route admin)
    RT --> P : 403
  else portée suffisante
    RT --> P : 200 + données
  end
end
G -> J : nom de la clé, route, code, motif\n(jamais le secret)
AD -> K : révoquer la clé
note over G, K : le trousseau est relu dès qu'il change :\nla révocation vaut à la requête suivante
'''

for nom, corps in D.items():
    (ICI / f"{nom}.puml").write_text("@startuml\n" + STYLE + corps.strip() + "\n@enduml\n", encoding="utf-8")
    print(nom)
