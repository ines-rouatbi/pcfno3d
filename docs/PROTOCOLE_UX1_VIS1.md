# Protocole de mesure UX1 et VIS1 (cahier v12, § 9.3)

*UX1 — compréhension : « lors d'un test avec 5 utilisateurs non spécialistes, au moins 4 trouvent la réponse
à une question métier en moins de 30 secondes ». VIS1 — tableau de bord : « les pages vue d'ensemble, vent 3D
et fiabilité affichent une reconstruction et ses indicateurs en moins de 3 secondes ».
Ces deux critères ne se mesurent pas par le code : ce protocole fixe la façon de les mesurer, avant la
mesure, pour que le résultat ne dépende pas de qui l'organise.*

## Préparation

- Le service tourne (notebook de service) et atom.zelqane.com répond ; noter la date, l'heure et le modèle
  affiché par le bandeau de la page.
- Un ordinateur portable courant et un téléphone ; navigateur à jour ; connexion ordinaire (pas de câble
  de bureau si les utilisateurs réels sont sur réseau mobile).
- Ouvrir `https://atom.zelqane.com/?test_ux=1` : un panneau « Test UX1 / VIS1 » apparaît en bas à gauche.
  Il chronomètre chaque tâche et les temps d'affichage, garde les résultats dans ce navigateur et les
  exporte en CSV. Rien n'est envoyé au service.

## Participants

Cinq personnes **non spécialistes** du vent ni de l'application : par exemple deux agriculteurs, un
technicien, deux personnes du grand public. Aucune ne doit avoir vu la page avant. Noter pour chacune un
identifiant anonyme (P1 à P5) et son profil, jamais son nom dans le fichier exporté.

## Déroulement (UX1)

1. Présentation en une phrase : « C'est une application qui dit le vent dans le Grand Tunis ; je vais vous
   poser cinq questions ; répondez en utilisant la page. » Aucune démonstration.
2. Pour chaque tâche, l'animateur clique **Démarrer**, lit la question à voix haute, ne guide pas. La personne
   répond oralement ; l'animateur clique **Trouvé** dès la bonne réponse, **Abandon** après 90 s ou sur abandon.
3. Les cinq tâches, dans cet ordre :
   1. « Peut-on traiter les vignes de Mornag demain matin ? » (espace agriculteur, J+1)
   2. « D'où vient le vent à Nabeul maintenant, et à quelle vitesse ? » (Mon vent)
   3. « Y a-t-il une alerte de vent fort aujourd'hui ? » (Alertes)
   4. « Combien d'eau faut-il pour 2 hectares de tomates aujourd'hui ? » (espace agriculteur)
   5. « Le vent est-il plus fort à 100 m qu'au sol au-dessus de Djebel Ressas ? » (Carte 3D ou profil)
4. **Verdict** : une personne réussit si elle trouve la bonne réponse en moins de 30 s à **la tâche 1**
   (question métier de référence du cahier). UX1 est conforme si au moins 4 personnes sur 5 réussissent.
   Les tâches 2 à 5 sont rapportées à part (taux de réussite et temps médian), pour l'amélioration.

## Mesure VIS1

Le panneau enregistre, à chaque ouverture des vues **Tableau de bord**, **Carte 3D** et **Fiabilité**, le temps
entre le clic et l'affichage des indicateurs (fin du rendu de la vue), ainsi que le temps de chargement de
la journée temps réel (réponse de `/api/atom-wind/direct`). VIS1 est conforme si, sur 10 ouvertures de chaque
vue après le premier chargement de la journée, le 90e centile reste sous 3 s. Le premier calcul de la journée
par le modèle (jusqu'à une minute, Colab) est rapporté à part : il n'est pas un affichage.

## Rapport

Exporter le CSV du panneau (bouton **Exporter**), le joindre au rapport avec : date, modèle, appareil, réseau,
profils P1 à P5, verdict UX1 (n / 5 sous 30 s à la tâche 1), temps médian par tâche, verdict VIS1 (90e centile
par vue). Les résultats, favorables ou non, sont rapportés tels quels.
