# Weather Max Predictor

Weather forecast engine tuned for weather prediction market workflows.

It now blends three independent public forecast sources:
- Open-Meteo
- NOAA/NWS
- MET Norway (met.no)

The script aligns forecasts to each airport's local day, applies observation anchoring, adapts source weights from rolling error history, and outputs fair-value probabilities for configured max-temperature market lines.

## Quick Start

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

3. Run:

```bash
python3 main.py
```

## What The Script Outputs

- Recent backtest panel (free archive-based verification):
	- MAE and RMSE on predicted daily max
	- Brier score on market probabilities
	- per-airport reliability summary
- Daily scoreboard across airports
- Per-airport prediction panel:
	- predicted max/min/mean
	- uncertainty sigma
	- confidence level and score
	- source agreement spread
- Source diagnostics table:
	- dynamic weight
	- bias adjustment
	- nowcast error
	- rolling historical MAE
- Market lines table:
	- P(Over), P(Under)
	- fair value in cents for each side

## Configure For Your Markets

Edit `AIRPORTS` in `main.py`:
- `tz`: airport local timezone used for day slicing
- `market_lines`: thresholds to evaluate for max-temp contracts

Example:

```python
"KXYZ": {
		"lat": 00.000,
		"lon": -00.000,
		"city": "Example City",
		"tz": "America/New_York",
		"market_lines": [72, 75, 78],
}
```

## Adaptive History

The script stores source error history in `forecast_performance.json` and updates it on each run.

If you want to reset model memory, delete that file.

### What Is `forecast_performance.json`?

It is a local state file that helps the model learn source reliability over time.

It stores:
- Per-airport source error lists (Open-Meteo, NWS, MET Norway)
- Last updated timestamp

It is not required for first run, and it is regenerated automatically.

### What Is `forecast_runs.jsonl`?

It is a line-by-line run history used for automatic backtesting.

Each line stores one airport prediction snapshot from a run:
- issued timestamp
- target day
- predicted max/min/mean/sigma
- market probabilities by line

On each run, the script compares recent predictions to observed max temperatures from the free Open-Meteo archive API and shows model accuracy metrics.

## Notes

- Terminal UI is powered by `rich`.
- If `pip` reports `externally-managed-environment`, use the local `.venv` above.
