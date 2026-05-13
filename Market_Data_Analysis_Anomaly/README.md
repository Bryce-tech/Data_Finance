# Market Data Quality — Anomaly Detection

## Overview

This project builds a simple Python pipeline for listed market data quality control.
It loads OHLCV data, applies rule-based checks, computes financial features, detects atypical observations with Isolation Forest, and exports an incident table.
The focus is data reliability, operational risk reduction, and reviewable control outputs.

## Use case

The project simulates a quality control pipeline for listed market data. It is designed around common checks used before market data is consumed by reporting, risk, PnL, or reference data processes.

## Pipeline

- data ingestion;
- instrument reference;
- quality checks;
- feature engineering;
- anomaly detection;
- incident table;
- exports.

## Quality checks

- missing prices;
- non-positive prices;
- OHLC inconsistencies;
- close outside high-low range;
- missing or non-positive volume;
- stale prices;
- extreme returns.

## Machine learning approach

Isolation Forest is used to identify observations with unusual combinations of price, volatility, volume, and quality-score features.
The model helps prioritize atypical records for review. It does not replace explicit business rules or source validation.

## Outputs

- `outputs/market_data_scored.csv`;
- `outputs/market_data_incidents.csv`;
- `outputs/instrument_reference.csv`.

## Project structure

```text
.
├── market_data_quality_anomaly_detection.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── outputs/
```

## Installation

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
jupyter notebook
```

Open `market_data_quality_anomaly_detection.ipynb` and run the cells from top to bottom.

## Limitations

- `yfinance` is used as a demonstration data source;
- thresholds should be calibrated by asset class, liquidity, currency, and source;
- the project does not use real incident labels;
- in production, controls should be historized and false positives should be monitored.
