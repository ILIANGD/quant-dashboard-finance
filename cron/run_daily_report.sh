#!/bin/bash

# =========================================================
# Wrapper script to run the daily report via Cron
# =========================================================

# 1. Se placer dans le dossier racine du projet (ajuste le chemin si besoin lors du déploiement)
# L'astuce ici est de récupérer le dossier parent du dossier 'cron' où se trouve ce script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
cd "$DIR"

# 2. Activer l'environnement virtuel (adapter le nom du dossier, ex: .venv, venv, env)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found."
    exit 1
fi

# 3. Ajouter le dossier courant au PYTHONPATH pour que 'import services...' fonctionne
export PYTHONPATH=$PYTHONPATH:.

# 4. Lancer le script Python
python cron/daily_report.py

# 5. Désactiver l'environnement (bonnes pratiques)
deactivate
