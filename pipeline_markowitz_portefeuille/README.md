# Pipeline d'optimisation de portefeuille Markowitz

Ce projet contient un notebook Python consacré à l'optimisation de portefeuille moyenne-variance. Il charge des prix d'actifs, estime les paramètres de rendement et de risque, optimise plusieurs allocations et compare les résultats avec des visualisations simples.

## Contenu

- Chargement de données via `yfinance`, avec fallback synthétique.
- Calcul des rendements logarithmiques, volatilités, covariances et corrélations.
- Portefeuilles équipondéré, minimum variance et maximum Sharpe.
- Contraintes de poids par actif et d'exposition sectorielle.
- Simulation de portefeuilles aléatoires et frontière efficiente.
- Mesures de risque: VaR, CVaR et drawdown.
- Simulation Monte Carlo de rendements corrélés.
- Backtest train/test hors échantillon.
- Shrinkage de covariance avec Ledoit-Wolf.

## Installation Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer le notebook

```bash
jupyter notebook 01_pipeline_optimisation_portefeuille_markowitz.ipynb
```

## Limites

Les résultats dépendent fortement de la période d'observation, des actifs retenus et de l'estimation des rendements attendus. Le backtest ne prend pas en compte les frais de transaction, la fiscalité, la liquidité ni les rééquilibrages périodiques.

