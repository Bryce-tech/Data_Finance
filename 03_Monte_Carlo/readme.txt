Environnement virtuel commun pour les projets 02, 03 et 04

Ce dossier contient un fichier requirements.txt commun aux projets :
- 02_Strategie_long_short
- 03_Monte_Carlo

Les notebooks sont disponibles à la lecture, ils sont déjà compilés. 

1. Creer l'environnement virtuel

Depuis la racine du projet, executer :

python -m venv .venv

2. Activer l'environnement virtuel

Sous PowerShell :

.\.venv\Scripts\Activate.ps1

Sous l'invite de commandes Windows (cmd) :

.\.venv\Scripts\activate.bat

3. Installer les dependances

Une fois l'environnement active, executer :

pip install -r requirements.txt

4. Desactiver l'environnement virtuel

Quand vous avez fini, executer :

deactivate
