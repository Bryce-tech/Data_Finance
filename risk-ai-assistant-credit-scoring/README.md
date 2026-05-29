# Risk AI Assistant — Scoring Crédit & Détection d'Anomalies

## Vue d'ensemble

Ce projet est un proof of concept compact pour une Direction des Risques bancaire. Il montre comment une chaîne data science simple peut passer de données de dossiers de crédit à une synthèse métier exploitable. La pipeline couvre les contrôles de qualité des données, le scoring supervisé du risque de crédit, la détection d'anomalies, l'interprétation du modèle et une explication contrôlée par règles. L'objectif n'est pas d'automatiser une décision de crédit, mais d'aider à prioriser les dossiers qui méritent une revue plus attentive.

## Objectif métier

Le POC aide une équipe Risques à identifier les dossiers de crédit nécessitant une revue manuelle. Il combine une probabilité de risque, un indicateur d'anomalie et des facteurs explicatifs lisibles afin de faciliter les échanges entre analystes risques, data scientists et responsables métier.

## Fonctionnalités principales

* Contrôles de qualité des données
* Scoring du risque de crédit
* Métriques de classification
* Détection d'anomalies
* Importance des variables
* Synthèse métier orientée risque
* Limites et gouvernance modèle

## Structure du projet

```text
risk-ai-assistant-credit-scoring/
│
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── simulated_credit_risk_dataset.csv
├── notebooks/
│   └── 01_risk_ai_assistant_credit_scoring.ipynb
└── src/
    └── risk_summary.py
```

## Rappels mathématiques

Le modèle de référence utilise une régression logistique. Elle estime la probabilité qu'un dossier soit risqué :

$$
p(y=1|x)=\frac{1}{1+e^{-z}}
$$

avec :

$$
z = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p
$$

Les principales métriques de classification sont :

$$
Precision=\frac{TP}{TP+FP}
$$

$$
Recall=\frac{TP}{TP+FN}
$$

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

La précision mesure la part des alertes qui sont réellement risquées. Le rappel mesure la part des dossiers risqués détectés par le modèle. Le F1-score équilibre ces deux dimensions.

## Résultats

Les résultats dépendent de l'échantillon généré et du découpage train/test. Le notebook présente :

* ROC-AUC
* Recall
* Precision
* F1-score
* Principaux facteurs de risque
* Anomalies détectées

Les facteurs de risque typiques sont un taux d'endettement élevé, des incidents de paiement récents, un défaut passé et un niveau d'épargne faible.

## Exécution

Créer un environnement virtuel :

```bash
python -m venv .venv
```

Activer l'environnement sous Windows PowerShell :

```bash
.venv\Scripts\Activate.ps1
```

Si PowerShell bloque l'exécution des scripts :

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer Jupyter :

```bash
jupyter notebook
```

Ouvrir le notebook :

```text
notebooks/01_risk_ai_assistant_credit_scoring.ipynb
```

## Limites

* Les données sont simulées et ne représentent pas un portefeuille réel.
* Le modèle n'a pas de validation réglementaire.
* Le score ne doit pas être utilisé comme décision automatique.
* Des biais peuvent exister dans les données simulées et dans une extension réelle.
* Un monitoring est nécessaire pour suivre le drift et la dégradation de performance.
* Une revue humaine reste indispensable pour les décisions sensibles.

## Extensions possibles

* Dashboard Streamlit
* Explications SHAP
* Monitoring modèle
* Détection de drift
* Synthèse contrôlée par LLM/RAG
* Connexion à des données internes de risque
