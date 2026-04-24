# Weather Max Predictor

Small utility that fetches hourly forecasts from Open-Meteo and NOAA/NWS and produces an adaptive ensemble forecast with a clear terminal UI.

## Quick start

1. Create and activate the project's virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

3. Run the script:

```bash
python3 main.py
```

## Notes
- The terminal UI uses `rich` for readable panels, tables, and spinners.
- The repository already ignores virtualenvs and caches via `.gitignore`.
- If you want less verbose output, edit `main.py` to change `console.print` styles or remove the detailed report print.

## Troubleshooting
- If `pip` reports an "externally-managed-environment" error, ensure you are using the local `.venv` (see step 1) or use `--user`/`pipx` as appropriate.

## Files
- `main.py` — main program
- `requirements.txt` — dependencies (`requests`, `rich`)

Feel free to ask me to add usage examples or CI steps.
