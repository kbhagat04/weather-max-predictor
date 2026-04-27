import json
import os
from datetime import date, datetime, timedelta, timezone
from math import erf, sqrt
from statistics import mean, stdev
from typing import List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# --- CONFIGURATION ---
AIRPORTS = {
    "KAUS": {
        "lat": 30.1944,
        "lon": -97.67,
        "city": "Austin",
        "tz": "America/Chicago",
        "market_lines": [88, 92, 96],
    },
    "KDFW": {
        "lat": 32.8968,
        "lon": -97.038,
        "city": "Dallas-Fort Worth",
        "tz": "America/Chicago",
        "market_lines": [90, 95, 100],
    },
    "KLGA": {
        "lat": 40.76,
        "lon": -73.86,
        "city": "New York",
        "tz": "America/New_York",
        "market_lines": [48, 52, 56],
    },
    "KSEA": {
        "lat": 47.448,
        "lon": -122.309,
        "city": "Seattle",
        "tz": "America/Los_Angeles",
        "market_lines": [58, 62, 66],
    },
}

FORECAST_HISTORY_FILE = "forecast_performance.json"
FORECAST_RUNS_FILE = "forecast_runs.jsonl"
MAX_HISTORY = 60
BACKTEST_LOOKBACK_DAYS = 14

AIRPORT_STYLES = {
    "KAUS": "bright_magenta",
    "KDFW": "bright_yellow",
    "KLGA": "bright_blue",
    "KSEA": "bright_cyan",
}

SOURCE_STYLES = {
    "openmeteo": "green",
    "nws": "cyan",
    "metno": "magenta",
}


# --- NETWORK LAYER ---
def build_http_session() -> requests.Session:
    """Build an HTTP session with retries for weather APIs."""
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "weather-market-model/2.0"})
    return session


# --- TIME / SERIES HELPERS ---
def parse_forecast_time(value: str) -> Optional[datetime]:
    """Parse common ISO datetime strings used by forecast providers."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def local_forecast_day(tz_name: str) -> datetime.date:
    """Today's date in the airport's local timezone."""
    return datetime.now(ZoneInfo(tz_name)).date()


def filter_series_to_day(times: List[str], values: List[float], target_day, tz_name: str) -> Tuple[List[str], List[float]]:
    """Filter hourly forecast values so max/min are computed on one local calendar day."""
    filtered_times: List[str] = []
    filtered_values: List[float] = []

    for time_value, value in zip(times, values):
        dt = parse_forecast_time(time_value)
        if dt is None:
            continue
        local_dt = dt.astimezone(ZoneInfo(tz_name))
        if local_dt.date() == target_day:
            filtered_times.append(time_value)
            filtered_values.append(value)

    # Fallback to original if filtering is too aggressive.
    if len(filtered_values) < 6:
        return list(times), list(values)
    return filtered_times, filtered_values


def interpolate_data(data: List[float], target_length: int) -> List[float]:
    """Linear interpolation to align each source to the same hourly length."""
    if not data:
        return []
    if target_length <= 1:
        return [data[0]]
    if len(data) == target_length:
        return data

    ratio = (len(data) - 1) / (target_length - 1)
    result: List[float] = []
    for i in range(target_length):
        idx = i * ratio
        low = int(idx)
        high = min(low + 1, len(data) - 1)
        weight = idx - low
        value = data[low] * (1 - weight) + data[high] * weight
        result.append(value)
    return result


def normalize_weights(raw_weights: List[float]) -> List[float]:
    """Return non-negative weights that sum to 1."""
    cleaned = [max(0.0, float(w)) for w in raw_weights]
    total = sum(cleaned)
    if total <= 0:
        return [1.0 / len(cleaned)] * len(cleaned)
    return [w / total for w in cleaned]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


# --- MARKET MATH HELPERS ---
def normal_cdf(x: float) -> float:
    """CDF for standard normal distribution."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def prob_over_line(mu: float, sigma: float, line: float) -> float:
    """Probability that temperature max ends above a given line."""
    sigma = max(0.8, sigma)
    z = (line - mu) / sigma
    return clamp(1.0 - normal_cdf(z), 0.0, 1.0)


def sparkline(values: List[float], width: int = 24) -> str:
    """Compact hourly trend chart for terminal output."""
    if not values:
        return "N/A"
    blocks = "▁▂▃▄▅▆▇█"
    if len(values) > width:
        values = interpolate_data(values, width)
    low = min(values)
    high = max(values)
    if high == low:
        return blocks[3] * len(values)
    chars = []
    for value in values:
        idx = int((value - low) / (high - low) * (len(blocks) - 1))
        chars.append(blocks[idx])
    return "".join(chars)


def ratio_bar(ratio: float, width: int = 12) -> str:
    """Simple progress bar for weights/probabilities."""
    ratio = clamp(ratio, 0.0, 1.0)
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


def parse_ymd(value: str) -> Optional[date]:
    """Parse YYYY-MM-DD to a date object."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def brier_score(probability: float, outcome: int) -> float:
    """Brier score for binary event calibration."""
    p = clamp(probability, 0.0, 1.0)
    o = 1.0 if outcome else 0.0
    return (p - o) ** 2


# --- HISTORY / ADAPTIVE WEIGHTS ---
def load_history() -> dict:
    """Load source error history used for adaptive source weighting."""
    if not os.path.exists(FORECAST_HISTORY_FILE):
        return {"airports": {}, "updated_at": None}

    try:
        with open(FORECAST_HISTORY_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {"airports": {}, "updated_at": None}

    if "airports" in raw:
        return raw

    # Backward compatibility with previous schema.
    converted = {"airports": {}, "updated_at": raw.get("timestamp")}
    converted["airports"]["GLOBAL"] = {
        "source_errors": {
            "openmeteo": raw.get("openmeteo_errors", []),
            "nws": raw.get("nws_errors", []),
            "metno": [],
        }
    }
    return converted


def save_history(history: dict) -> None:
    """Persist error history for future runs."""
    history["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with open(FORECAST_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def get_source_errors(history: dict, airport_code: str, source: str) -> List[float]:
    airport_data = history.get("airports", {}).get(airport_code, {})
    source_errors = airport_data.get("source_errors", {}).get(source, [])
    if source_errors:
        return source_errors

    # Fallback to global history for older files.
    global_data = history.get("airports", {}).get("GLOBAL", {})
    return global_data.get("source_errors", {}).get(source, [])


def append_source_error(history: dict, airport_code: str, source: str, error_value: float) -> None:
    airport_bucket = history.setdefault("airports", {}).setdefault(airport_code, {"source_errors": {}})
    errors = airport_bucket.setdefault("source_errors", {}).setdefault(source, [])
    errors.append(abs(float(error_value)))
    if len(errors) > MAX_HISTORY:
        del errors[:-MAX_HISTORY]


def source_weight_from_errors(history_errors: List[float], nowcast_abs_error: Optional[float]) -> float:
    """Translate error magnitudes into reliability score (higher is better)."""
    historical_mae = mean(history_errors) if history_errors else 2.5
    nowcast_mae = nowcast_abs_error if nowcast_abs_error is not None else 2.5

    hist_component = 1.0 / (historical_mae + 0.25)
    now_component = 1.0 / (nowcast_mae + 0.25)
    return 0.7 * hist_component + 0.3 * now_component


def append_forecast_runs(run_timestamp: str, results: List[dict]) -> None:
    """Append one record per airport forecast run for later backtesting."""
    if not results:
        return

    with open(FORECAST_RUNS_FILE, "a", encoding="utf-8") as f:
        for result in results:
            record = {
                "issued_at": run_timestamp,
                "airport_code": result["airport_code"],
                "city": result["city"],
                "target_day": result["target_day"],
                "pred_max": result["pred_max"],
                "pred_min": result["pred_min"],
                "pred_mean": result["pred_mean"],
                "sigma_max": result["sigma_max"],
                "confidence_score": result["confidence_score"],
                "market": [
                    {
                        "line": row["line"],
                        "p_over": row["p_over"],
                    }
                    for row in result.get("market", [])
                ],
            }
            f.write(json.dumps(record) + "\n")


def load_forecast_runs(max_records: int = 4000) -> List[dict]:
    """Read recent forecast run records from jsonl storage."""
    if not os.path.exists(FORECAST_RUNS_FILE):
        return []

    try:
        with open(FORECAST_RUNS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []

    if max_records > 0:
        lines = lines[-max_records:]

    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return records


def fetch_observed_daily_max_openmeteo(
    session: requests.Session,
    lat: float,
    lon: float,
    target_day: date,
    tz_name: str,
) -> Optional[float]:
    """Fetch observed daily max temperature from Open-Meteo archive."""
    day_str = target_day.strftime("%Y-%m-%d")
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={day_str}&end_date={day_str}"
        "&daily=temperature_2m_max"
        "&temperature_unit=fahrenheit"
        f"&timezone={tz_name}"
    )
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        values = daily.get("temperature_2m_max", [])
        if not values:
            return None
        return float(values[0])
    except Exception:
        return None


def evaluate_recent_performance(session: requests.Session, lookback_days: int = BACKTEST_LOOKBACK_DAYS) -> dict:
    """Backtest recent predictions against observed daily max from a free archive API."""
    runs = load_forecast_runs()
    if not runs:
        return {"overall": None, "airports": {}}

    per_airport = {}

    for airport_code, airport_info in AIRPORTS.items():
        today_local = local_forecast_day(airport_info["tz"])
        start_day = today_local - timedelta(days=lookback_days)
        end_day = today_local - timedelta(days=1)
        if end_day < start_day:
            continue

        # Keep only the latest run per target day.
        latest_by_day = {}
        for record in runs:
            if record.get("airport_code") != airport_code:
                continue
            target_day = parse_ymd(record.get("target_day", ""))
            if target_day is None or not (start_day <= target_day <= end_day):
                continue

            day_key = target_day.strftime("%Y-%m-%d")
            previous = latest_by_day.get(day_key)
            if previous is None or str(record.get("issued_at", "")) > str(previous.get("issued_at", "")):
                latest_by_day[day_key] = record

        abs_errors = []
        sq_errors = []
        brier_values = []

        for day_key, record in sorted(latest_by_day.items()):
            target_day = parse_ymd(day_key)
            if target_day is None:
                continue

            observed_max = fetch_observed_daily_max_openmeteo(
                session,
                airport_info["lat"],
                airport_info["lon"],
                target_day,
                airport_info["tz"],
            )
            if observed_max is None:
                continue

            pred_max = float(record.get("pred_max", 0.0))
            diff = pred_max - observed_max
            abs_errors.append(abs(diff))
            sq_errors.append(diff * diff)

            for row in record.get("market", []):
                try:
                    line = float(row.get("line"))
                    p_over = float(row.get("p_over"))
                except Exception:
                    continue
                outcome = 1 if observed_max > line else 0
                brier_values.append(brier_score(p_over, outcome))

        if abs_errors:
            rmse = sqrt(sum(sq_errors) / len(sq_errors))
            brier = (sum(brier_values) / len(brier_values)) if brier_values else None
            per_airport[airport_code] = {
                "samples": len(abs_errors),
                "mae": sum(abs_errors) / len(abs_errors),
                "rmse": rmse,
                "brier": brier,
                "market_samples": len(brier_values),
            }

    if not per_airport:
        return {"overall": None, "airports": {}}

    all_abs = [stats["mae"] * stats["samples"] for stats in per_airport.values()]
    all_samples = [stats["samples"] for stats in per_airport.values()]
    all_rmse_num = [((stats["rmse"] ** 2) * stats["samples"]) for stats in per_airport.values()]
    all_market_num = [
        (stats["brier"] * stats["market_samples"]) for stats in per_airport.values() if stats["brier"] is not None
    ]
    all_market_samples = [
        stats["market_samples"] for stats in per_airport.values() if stats["brier"] is not None
    ]

    total_samples = sum(all_samples)
    total_mae = (sum(all_abs) / total_samples) if total_samples else None
    total_rmse = sqrt(sum(all_rmse_num) / total_samples) if total_samples else None
    total_market_samples = sum(all_market_samples)
    total_brier = (sum(all_market_num) / total_market_samples) if total_market_samples else None

    return {
        "overall": {
            "samples": total_samples,
            "mae": total_mae,
            "rmse": total_rmse,
            "brier": total_brier,
            "market_samples": total_market_samples,
            "lookback_days": lookback_days,
        },
        "airports": per_airport,
    }


# --- SOURCE FETCHERS ---
def fetch_open_meteo(session: requests.Session, lat: float, lon: float) -> Optional[dict]:
    """Open-Meteo hourly forecast in Fahrenheit."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m"
        "&forecast_days=2&temperature_unit=fahrenheit&timezone=UTC"
    )
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()["hourly"]
        return {
            "source": "openmeteo",
            "temps": [float(v) for v in data.get("temperature_2m", [])],
            "times": data.get("time", []),
            "wind": [float(v) for v in data.get("wind_speed_10m", [])],
        }
    except Exception:
        return None


def fetch_nws(session: requests.Session, lat: float, lon: float) -> Optional[dict]:
    """NWS hourly forecast in Fahrenheit."""
    try:
        points = session.get(f"https://api.weather.gov/points/{lat},{lon}", timeout=30)
        points.raise_for_status()
        forecast_url = points.json()["properties"]["forecastHourly"]

        forecast = session.get(forecast_url, timeout=30)
        forecast.raise_for_status()
        periods = forecast.json()["properties"]["periods"]

        temps = [float(p["temperature"]) for p in periods][:72]
        times = [p.get("startTime", "") for p in periods][:72]
        return {
            "source": "nws",
            "temps": temps,
            "times": times,
            "wind": [],
        }
    except Exception:
        return None


def fetch_metno(session: requests.Session, lat: float, lon: float) -> Optional[dict]:
    """MET Norway hourly forecast converted from Celsius to Fahrenheit."""
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
    headers = {"User-Agent": "weather-market-model/2.0 (contact: local-script)"}
    try:
        resp = session.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        timeseries = resp.json()["properties"]["timeseries"][:72]

        temps_f = []
        times = []
        for row in timeseries:
            celsius = row["data"]["instant"]["details"].get("air_temperature")
            if celsius is None:
                continue
            temps_f.append((float(celsius) * 9 / 5) + 32)
            times.append(row.get("time", ""))

        return {
            "source": "metno",
            "temps": temps_f,
            "times": times,
            "wind": [],
        }
    except Exception:
        return None


def fetch_current_obs_openmeteo(session: requests.Session, lat: float, lon: float) -> Optional[float]:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current=temperature_2m&temperature_unit=fahrenheit"
    )
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        return float(resp.json().get("current", {}).get("temperature_2m"))
    except Exception:
        return None


def fetch_current_obs_nws(session: requests.Session, lat: float, lon: float) -> Optional[float]:
    try:
        points = session.get(f"https://api.weather.gov/points/{lat},{lon}", timeout=25)
        points.raise_for_status()
        stations_url = points.json()["properties"]["observationStations"]

        stations = session.get(stations_url, timeout=25)
        stations.raise_for_status()
        station_id = stations.json()["features"][0]["id"]

        latest = session.get(f"{station_id}/observations/latest", timeout=25)
        latest.raise_for_status()
        temp_c = latest.json()["properties"]["temperature"]["value"]
        if temp_c is None:
            return None
        return (float(temp_c) * 9 / 5) + 32
    except Exception:
        return None


# --- FORECAST ENGINE ---
def build_airport_forecast(
    airport_code: str,
    airport_info: dict,
    source_data: List[dict],
    current_obs: Optional[float],
    history: dict,
) -> Optional[dict]:
    """Create a day-aligned ensemble forecast with source diagnostics."""
    target_day = local_forecast_day(airport_info["tz"])

    usable_sources = []
    for item in source_data:
        if not item or not item.get("temps"):
            continue

        day_times, day_temps = filter_series_to_day(
            item.get("times", []),
            item.get("temps", []),
            target_day,
            airport_info["tz"],
        )
        if not day_temps:
            continue

        nowcast_error = None
        bias = 0.0
        if current_obs is not None:
            nowcast_error = abs(day_temps[0] - current_obs)
            # Anchor each source to current conditions but cap adjustment.
            bias = clamp(current_obs - day_temps[0], -7.0, 7.0)

        corrected = [t + bias for t in day_temps]
        history_errors = get_source_errors(history, airport_code, item["source"])
        score = source_weight_from_errors(history_errors, nowcast_error)

        usable_sources.append(
            {
                "source": item["source"],
                "times": day_times,
                "temps": corrected,
                "raw_temps": day_temps,
                "bias": bias,
                "nowcast_error": nowcast_error,
                "history_mae": mean(history_errors) if history_errors else None,
                "score": score,
            }
        )

    if len(usable_sources) < 2:
        return None

    target_len = max(len(s["temps"]) for s in usable_sources)
    for source in usable_sources:
        source["aligned"] = interpolate_data(source["temps"], target_len)

    weights = normalize_weights([s["score"] for s in usable_sources])
    for source, weight in zip(usable_sources, weights):
        source["weight"] = weight

    hourly_blend = []
    for i in range(target_len):
        blended_value = sum(s["aligned"][i] * s["weight"] for s in usable_sources)
        hourly_blend.append(blended_value)

    source_maxes = [max(s["aligned"]) for s in usable_sources]
    max_spread = max(source_maxes) - min(source_maxes)

    pred_max = max(hourly_blend)
    pred_min = min(hourly_blend)
    pred_mean = mean(hourly_blend)

    spread_std = stdev(source_maxes) if len(source_maxes) > 1 else 1.0
    day_volatility = stdev(hourly_blend) if len(hourly_blend) > 2 else 2.0
    sigma_max = max(1.2, spread_std + 0.10 * day_volatility + 0.20 * max_spread)

    confidence_score = int(clamp(100.0 - (max_spread * 8.0 + spread_std * 6.0), 5.0, 95.0))
    if confidence_score >= 75:
        confidence_level = "HIGH"
    elif confidence_score >= 50:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    market_rows = []
    for line in airport_info.get("market_lines", []):
        p_over = prob_over_line(pred_max, sigma_max, line)
        p_under = 1.0 - p_over
        market_rows.append(
            {
                "line": float(line),
                "p_over": p_over,
                "p_under": p_under,
                "fair_over_cents": round(p_over * 100, 1),
                "fair_under_cents": round(p_under * 100, 1),
            }
        )

    return {
        "airport_code": airport_code,
        "city": airport_info["city"],
        "target_day": str(target_day),
        "pred_max": pred_max,
        "pred_min": pred_min,
        "pred_mean": pred_mean,
        "pred_range": pred_max - pred_min,
        "hourly": hourly_blend,
        "sigma_max": sigma_max,
        "confidence_score": confidence_score,
        "confidence_level": confidence_level,
        "source_spread": max_spread,
        "sources": usable_sources,
        "market": market_rows,
        "current_obs": current_obs,
    }


def update_history_with_nowcast_errors(history: dict, airport_code: str, forecast_result: dict) -> None:
    """Use current-observation anchor error as online calibration signal."""
    for source in forecast_result.get("sources", []):
        if source.get("nowcast_error") is not None:
            append_source_error(history, airport_code, source["source"], source["nowcast_error"])


# --- UI ---
def render_airport_report(console: Console, result: dict) -> None:
    title = f"{result['airport_code']} - {result['city']}"
    airport_style = AIRPORT_STYLES.get(result["airport_code"], "white")

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Forecast Day", result["target_day"])
    summary.add_row("Predicted Max", f"{result['pred_max']:.1f}F")
    summary.add_row("Predicted Min", f"{result['pred_min']:.1f}F")
    summary.add_row("Predicted Mean", f"{result['pred_mean']:.1f}F")
    summary.add_row("Daily Range", f"{result['pred_range']:.1f}F")
    summary.add_row("Uncertainty (sigma)", f"{result['sigma_max']:.2f}F")
    summary.add_row("Confidence", f"{result['confidence_level']} ({result['confidence_score']}/100)")
    summary.add_row("Source Agreement", f"{result['source_spread']:.1f}F spread")
    summary.add_row("Sources Used", str(len(result["sources"])))
    if result.get("current_obs") is not None:
        summary.add_row("Current Observation", f"{result['current_obs']:.1f}F")
    summary.add_row("Hourly Trend", sparkline(result["hourly"]))

    console.print(Panel(summary, title=title, box=box.ROUNDED, expand=False, border_style=airport_style))

    src_table = Table(title="Source Diagnostics", box=box.SIMPLE_HEAVY, title_style=airport_style)
    src_table.add_column("Source", no_wrap=True)
    src_table.add_column("Weight", justify="right")
    src_table.add_column("Weight Bar", no_wrap=True)
    src_table.add_column("Bias", justify="right")
    src_table.add_column("Nowcast Err", justify="right")
    src_table.add_column("Hist MAE", justify="right")
    src_table.add_column("Day Max", justify="right")
    src_table.add_column("Day Min", justify="right")
    for src in result["sources"]:
        now_err = "-" if src.get("nowcast_error") is None else f"{src['nowcast_error']:.2f}"
        hist_mae = "-" if src.get("history_mae") is None else f"{src['history_mae']:.2f}"
        source_style = SOURCE_STYLES.get(src["source"], "white")
        src_table.add_row(
            f"[{source_style}]{src['source']}[/]",
            f"{src['weight']:.2f}",
            ratio_bar(src["weight"]),
            f"{src['bias']:+.2f}",
            now_err,
            hist_mae,
            f"{max(src['aligned']):.1f}",
            f"{min(src['aligned']):.1f}",
        )
    console.print(src_table)

    market_table = Table(title="Market Lines (Fair Value View)", box=box.SIMPLE, title_style=airport_style)
    market_table.add_column("Line (Max Temp F)", justify="right")
    market_table.add_column("P(Over)", justify="right")
    market_table.add_column("P(Under)", justify="right")
    market_table.add_column("P(Over) Bar")
    market_table.add_column("Fair Over", justify="right")
    market_table.add_column("Fair Under", justify="right")
    for row in result["market"]:
        market_table.add_row(
            f"{row['line']:.1f}",
            f"{row['p_over']*100:.1f}%",
            f"{row['p_under']*100:.1f}%",
            ratio_bar(row["p_over"]),
            f"{row['fair_over_cents']:.1f}c",
            f"{row['fair_under_cents']:.1f}c",
        )
    console.print(market_table)
    console.print()


def render_overall_summary(console: Console, results: List[dict]) -> None:
    """Cross-airport scoreboard for quick scan before diving into details."""
    table = Table(title="Daily Forecast Scoreboard", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Airport")
    table.add_column("Pred Max", justify="right")
    table.add_column("Pred Min", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Sigma", justify="right")
    table.add_column("Best Over Signal", justify="right")

    for result in results:
        best_line = max(result["market"], key=lambda m: abs(m["p_over"] - 0.5)) if result["market"] else None
        signal = "-"
        if best_line:
            side = "Over" if best_line["p_over"] >= 0.5 else "Under"
            prob = best_line["p_over"] if side == "Over" else best_line["p_under"]
            signal = f"{side} {best_line['line']:.0f} ({prob*100:.0f}%)"
        airport_style = AIRPORT_STYLES.get(result["airport_code"], "white")
        table.add_row(
            f"[{airport_style}]{result['airport_code']}[/]",
            f"{result['pred_max']:.1f}F",
            f"{result['pred_min']:.1f}F",
            f"{result['confidence_level']} {result['confidence_score']}",
            f"{result['sigma_max']:.2f}",
            signal,
        )
    console.print(table)
    console.print()


def render_backtest_summary(console: Console, backtest: dict) -> None:
    """Render recent model performance from recorded forecast runs."""
    overall = backtest.get("overall") if backtest else None
    if not overall:
        console.print(
            Panel(
                Text("Backtest: not enough settled history yet. Keep running the script daily to build metrics."),
                title="Model Performance",
                style="yellow",
            )
        )
        console.print()
        return

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Lookback", f"{overall['lookback_days']} days")
    summary.add_row("Forecast Samples", str(overall["samples"]))
    summary.add_row("MAE (Max Temp)", f"{overall['mae']:.2f}F")
    summary.add_row("RMSE (Max Temp)", f"{overall['rmse']:.2f}F")
    if overall.get("brier") is not None:
        summary.add_row("Brier (Market Prob)", f"{overall['brier']:.4f}")
        summary.add_row("Market Samples", str(overall.get("market_samples", 0)))
    console.print(Panel(summary, title="Recent Backtest", border_style="green", box=box.ROUNDED, expand=False))

    airport_table = Table(title="Per-Airport Backtest", box=box.SIMPLE)
    airport_table.add_column("Airport")
    airport_table.add_column("Samples", justify="right")
    airport_table.add_column("MAE", justify="right")
    airport_table.add_column("RMSE", justify="right")
    airport_table.add_column("Brier", justify="right")

    for airport_code in sorted(backtest.get("airports", {}).keys()):
        stats = backtest["airports"][airport_code]
        brier = "-" if stats.get("brier") is None else f"{stats['brier']:.4f}"
        airport_style = AIRPORT_STYLES.get(airport_code, "white")
        airport_table.add_row(
            f"[{airport_style}]{airport_code}[/]",
            str(stats["samples"]),
            f"{stats['mae']:.2f}F",
            f"{stats['rmse']:.2f}F",
            brier,
        )

    console.print(airport_table)
    console.print()


# --- MAIN ---
def main() -> None:
    console = Console()
    session = build_http_session()
    run_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.rule("Weather Market Forecast Engine")
    console.print(f"Generated: {timestamp}", style="dim")
    console.print("Model: Adaptive 3-source blend with observation anchoring", style="dim")
    console.print()

    history = load_history()
    all_results: List[dict] = []

    for airport_code, airport_info in AIRPORTS.items():
        with console.status(f"Collecting and blending data for {airport_code}...", spinner="dots"):
            sources = [
                fetch_open_meteo(session, airport_info["lat"], airport_info["lon"]),
                fetch_nws(session, airport_info["lat"], airport_info["lon"]),
                fetch_metno(session, airport_info["lat"], airport_info["lon"]),
            ]

            obs_candidates = [
                fetch_current_obs_openmeteo(session, airport_info["lat"], airport_info["lon"]),
                fetch_current_obs_nws(session, airport_info["lat"], airport_info["lon"]),
            ]
            obs_values = [v for v in obs_candidates if v is not None]
            current_obs = mean(obs_values) if obs_values else None

            result = build_airport_forecast(
                airport_code=airport_code,
                airport_info=airport_info,
                source_data=sources,
                current_obs=current_obs,
                history=history,
            )

        if result is None:
            console.print(
                Panel(
                    Text(f"{airport_code} ({airport_info['city']}): insufficient valid source data."),
                    title="Data Warning",
                    style="red",
                )
            )
            continue

        update_history_with_nowcast_errors(history, airport_code, result)
        all_results.append(result)

    if all_results:
        append_forecast_runs(run_timestamp, all_results)
        backtest = evaluate_recent_performance(session)
        render_backtest_summary(console, backtest)
        render_overall_summary(console, all_results)
        for result in all_results:
            airport_style = AIRPORT_STYLES.get(result["airport_code"], "white")
            console.rule(f"[bold {airport_style}]{result['airport_code']} / {result['city']}[/]")
            render_airport_report(console, result)
    else:
        console.print(Panel(Text("No airport forecasts were produced."), style="red"))

    save_history(history)
    console.rule()
    console.print("Run complete. History, run log, and backtest metrics updated.", style="green")


if __name__ == "__main__":
    main()
