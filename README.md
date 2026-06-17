# Analyseur de Match Football

Application Python qui analyse une courte vidéo de match et génère des statistiques automatiques.

## Fonctionnalités

- Détection des **joueurs** et du **ballon** (YOLOv8)
- Suivi multi-joueurs avec ID unique
- Détection des **passes**, **tirs** et **possession**
- Stats par joueur : passes, tirs, touches, distance, possession
- Stats par équipe (gauche / droite du terrain)
- Interface **bureau** (fenêtre Windows, sans serveur web)
- Vidéo annotée en sortie

## Installation

```bash
cd "Projet fin d'année"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> Au premier lancement, YOLOv8 télécharge automatiquement le modèle `yolov8n.pt` (~6 Mo).

## Lancement

**Application bureau (recommandé) :**

```bash
python gui_app.py
```

Ou double-cliquez sur `Lancer.bat`.

> Ancienne version web (optionnelle) : `streamlit run app.py`

## Utilisation

1. Cliquez sur **Choisir une vidéo** (.mp4, .avi, .mov)
2. Ajustez les paramètres si besoin
3. Cliquez sur **Lancer l'analyse**
4. Consultez les onglets Résumé / Équipes / Joueurs / Événements
5. Ouvrez la vidéo annotée avec le bouton dédié

## Conseils pour de meilleurs résultats

- Vidéo de **30 secondes à 2 minutes**
- Caméra **stable** (pas de zoom pendant le jeu)
- Joueurs **visibles en entier**
- Ballon **visible** autant que possible
- Résolution **720p** minimum

## Structure du projet

```
├── gui_app.py              # Interface bureau (principale)
├── app.py                  # Ancienne interface Streamlit (optionnelle)
├── analyzer/
│   ├── detector.py         # Détection YOLO
│   ├── tracker.py          # Suivi des joueurs
│   ├── events.py           # Passes, tirs, possession
│   ├── stats.py            # Agrégation des stats
│   └── video_analyzer.py   # Pipeline principal
└── requirements.txt
```

## Limitations

Ce projet est un **prototype éducatif**. La précision dépend fortement de la qualité vidéo. Les équipes sont estimées par position (gauche/droite). Pour une analyse professionnelle, il faudrait des modèles spécialisés football et plusieurs caméras.
