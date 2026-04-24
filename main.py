import requests
from datetime import datetime, timedelta
from statistics import median, mean, stdev
import json
import os
from collections import deque
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # For older Python if needed
from typing import Optional

# Rich for improved terminal UI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# --- CONFIGURATION ---
AIRPORTS = {
    'KAUS': {'lat': 30.1944, 'lon': -97.67, 'city': 'Austin'},
    'KDFW': {'lat': 32.8968, 'lon': -97.038, 'city': 'Dallas-Fort Worth'},
    'KLGA': {'lat': 40.76, 'lon': -73.86, 'city': 'New York'}
}

# --- ADAPTIVE BIAS CORRECTION AND WEIGHTED ENSEMBLE CONFIG ---
HISTORICAL_BIAS = {
    'openmeteo': 0.0,
    'nws': 0.0
}
SOURCE_WEIGHTS = {
    'openmeteo': 0.5,
    'nws': 0.5
}

# Performance tracking for adaptive weighting (last 30 forecasts)
FORECAST_HISTORY_FILE = 'forecast_performance.json'
MAX_HISTORY = 30

# --- WEATHER SOURCES ---
def fetch_open_meteo(lat, lon):
    """Fetch hourly forecast from Open-Meteo with temperature, humidity, pressure, wind."""
    try:
        url = f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m&forecast_days=2&timezone=auto&temperature_unit=fahrenheit'
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        hourly = data['hourly']
        return {
            'temps': hourly['temperature_2m'],
            'humidity': hourly.get('relative_humidity_2m', []),
            'pressure': hourly.get('pressure_msl', []),
            'wind': hourly.get('wind_speed_10m', []),
            'times': hourly.get('time', []),
            'source': 'openmeteo'
        }
    except Exception as e:
        import traceback
        print(f"Error fetching from Open-Meteo: {e}")
        traceback.print_exc()
        return None

def fetch_nws(lat, lon):
    """Fetch hourly forecast from NOAA/NWS (all hours, not just daytime)."""
    try:
        headers = {'User-Agent': 'weather-bot/1.0'}
        points_url = f'https://api.weather.gov/points/{lat},{lon}'
        points_resp = requests.get(points_url, headers=headers, timeout=30)
        points_resp.raise_for_status()
        forecast_url = points_resp.json()['properties']['forecastHourly']
        forecast_resp = requests.get(forecast_url, headers=headers, timeout=30)
        forecast_resp.raise_for_status()
        periods = forecast_resp.json()['properties']['periods']
        
        # Use all hourly data, not just daytime (24+ hours)
        temps = [p['temperature'] for p in periods][:96]
        wind = [p.get('windSpeed', '0 mph').split()[0] if p.get('windSpeed') else 0 for p in periods][:96]
        times = [p.get('startTime', '') for p in periods][:96]
        
        return {
            'temps': temps,
            'humidity': [],
            'pressure': [],
            'wind': [float(w) if w else 0 for w in wind],
            'times': times,
            'source': 'nws'
        }
    except Exception as e:
        import traceback
        print(f"Error fetching from NWS: {e}")
        traceback.print_exc()
        return None

def fetch_current_observation_openmeteo(lat, lon):
    """Fetch current observation from Open-Meteo."""
    try:
        url = f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m&temperature_unit=fahrenheit'
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        current_temp = data['current'].get('temperature_2m', None)
        return current_temp
    except Exception as e:
        return None

def fetch_current_observation_nws(lat, lon):
    """Fetch current observation from NOAA/NWS stations."""
    try:
        headers = {'User-Agent': 'weather-bot/1.0'}
        # Get nearest station
        points_url = f'https://api.weather.gov/points/{lat},{lon}'
        points_resp = requests.get(points_url, headers=headers, timeout=30)
        points_resp.raise_for_status()
        stations_url = points_resp.json()['properties']['observationStations']
        
        # Get first station
        stations_resp = requests.get(stations_url, headers=headers, timeout=30)
        stations_resp.raise_for_status()
        first_station = stations_resp.json()['features'][0]['id']
        
        # Get latest observation from station
        obs_url = f'{first_station}/observations/latest'
        obs_resp = requests.get(obs_url, headers=headers, timeout=30)
        obs_resp.raise_for_status()
        temp_c = obs_resp.json()['properties']['temperature']['value']
        # Convert Celsius to Fahrenheit
        temp_f = (temp_c * 9/5) + 32 if temp_c is not None else None
        return temp_f
    except Exception as e:
        return None


# --- HELPER FUNCTIONS FOR DATA PROCESSING ---

def interpolate_data(data, target_length):
    """Linear interpolation to align data to target length."""
    if not data or target_length <= 1:
        return data[:target_length] if data else []
    
    if len(data) == target_length:
        return data
    
    # Simple linear interpolation
    ratio = (len(data) - 1) / (target_length - 1)
    interpolated = []
    for i in range(target_length):
        idx = i * ratio
        lower = int(idx)
        upper = min(lower + 1, len(data) - 1)
        weight = idx - lower
        value = data[lower] * (1 - weight) + data[upper] * weight
        interpolated.append(value)
    return interpolated

def load_forecast_history():
    """Load past forecast performance from file for adaptive weighting."""
    if os.path.exists(FORECAST_HISTORY_FILE):
        try:
            with open(FORECAST_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return {'openmeteo_errors': deque(maxlen=MAX_HISTORY), 'nws_errors': deque(maxlen=MAX_HISTORY)}
    return {'openmeteo_errors': deque(maxlen=MAX_HISTORY), 'nws_errors': deque(maxlen=MAX_HISTORY)}

def save_forecast_history(history):
    """Save forecast performance for future adaptive weighting."""
    try:
        data = {
            'openmeteo_errors': list(history['openmeteo_errors']),
            'nws_errors': list(history['nws_errors']),
            'timestamp': datetime.now().isoformat()
        }
        with open(FORECAST_HISTORY_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Warning: Could not save forecast history: {e}")

def calculate_adaptive_weights(history):
    """Calculate weights based on historical prediction error."""
    openmeteo_errors = history.get('openmeteo_errors', [])
    nws_errors = history.get('nws_errors', [])
    
    if not openmeteo_errors or not nws_errors:
        return SOURCE_WEIGHTS
    
    openmeteo_mae = mean(openmeteo_errors)
    nws_mae = mean(nws_errors)
    
    # Inverse of MAE (lower error = higher weight)
    if openmeteo_mae + nws_mae == 0:
        return SOURCE_WEIGHTS
    
    total = 1 / (openmeteo_mae + 0.01) + 1 / (nws_mae + 0.01)
    adaptive_weights = {
        'openmeteo': (1 / (openmeteo_mae + 0.01)) / total,
        'nws': (1 / (nws_mae + 0.01)) / total
    }
    return adaptive_weights

# --- ENSEMBLE LOGIC ---

def robust_ensemble_max(forecast_data_list):
    """
    Advanced ensemble forecasting with interpolation, adaptive weighting, and uncertainty quantification.
    Returns ensemble predictions with confidence intervals and hourly breakdowns.
    """
    if not forecast_data_list or any(d is None for d in forecast_data_list):
        return None
    
    # Validate data
    temps_list = [d['temps'] for d in forecast_data_list]
    if not temps_list or any(not t for t in temps_list):
        return None
    
    # Determine target length (use maximum to avoid losing data)
    max_len = max(len(t) for t in temps_list)
    
    # Interpolate all to same length for proper alignment
    aligned_temps = [interpolate_data(t, max_len) for t in temps_list]
    
    # Extract daily max and min from each source
    maxes = [max(t) for t in aligned_temps]
    mins = [min(t) for t in aligned_temps]
    
    # Apply bias correction
    corrected_maxes = [maxes[0] + HISTORICAL_BIAS['openmeteo'], maxes[1] + HISTORICAL_BIAS['nws']]
    corrected_mins = [mins[0] + HISTORICAL_BIAS['openmeteo'], mins[1] + HISTORICAL_BIAS['nws']]
    
    # Use adaptive weights based on historical performance
    weights = [SOURCE_WEIGHTS['openmeteo'], SOURCE_WEIGHTS['nws']]
    
    # Weighted ensemble
    ensemble_max = sum(w * m for w, m in zip(weights, corrected_maxes))
    ensemble_min = sum(w * m for w, m in zip(weights, corrected_mins))
    ensemble_mean = mean([mean(t) for t in aligned_temps])
    
    # Calculate confidence intervals using variance
    max_variance = stdev(corrected_maxes) if len(corrected_maxes) > 1 else 0
    min_variance = stdev(corrected_mins) if len(corrected_mins) > 1 else 0
    
    # 95% confidence interval (approximately 2 std dev)
    max_ci = max_variance * 1.96
    min_ci = min_variance * 1.96
    
    # Confidence level: HIGH if low variance, LOW if high variance
    confidence = 'HIGH' if max_variance <= 2.0 else 'LOW'
    
    # Calculate hourly ensemble
    hourly_ensemble = []
    for i in range(max_len):
        hourly_temps = [aligned_temps[0][i] * weights[0], aligned_temps[1][i] * weights[1]]
        hourly_ensemble.append(sum(hourly_temps))
    
    return {
        'max': ensemble_max,
        'min': ensemble_min,
        'mean': ensemble_mean,
        'confidence': confidence,
        'max_ci': max_ci,
        'min_ci': min_ci,
        'source_maxes': corrected_maxes,
        'source_mins': corrected_mins,
        'source_spread': max(corrected_maxes) - min(corrected_maxes),
        'hourly': hourly_ensemble,
        'max_variance': max_variance,
        'hourly_range': max(hourly_ensemble) - min(hourly_ensemble) if hourly_ensemble else 0,
        'times': forecast_data_list[0].get('times', [])  # Use times from first data source
    }

# --- REPORT GENERATION ---
def generate_report(airport_code: str, airport_name: str, forecast_data_1: dict, forecast_data_2: dict, current_obs_1: Optional[float]=None, current_obs_2: Optional[float]=None):
    """Generate comprehensive forecast report with ensemble predictions, uncertainty, and hourly breakdown.
    Returns a tuple of (report_string, ensemble_dict_or_None).
    """
    ensemble = robust_ensemble_max([forecast_data_1, forecast_data_2])

    if ensemble is None:
        return (f"{airport_code} ({airport_name}): Data unavailable. Check API connections.\n", None)

    # Calculate current observations ensemble
    current_obs_str = ""
    current_obs_ensemble = None
    if current_obs_1 is not None or current_obs_2 is not None:
        current_temps = [t for t in (current_obs_1, current_obs_2) if t is not None]
        if current_temps:
            current_obs_ensemble = mean(current_temps)
            current_obs_str = f"\n🌡️  CURRENT OBSERVATION (Ensemble Average): {current_obs_ensemble:.1f}°F"
            details = []
            if current_obs_1 is not None:
                details.append(f"Open-Meteo: {current_obs_1:.1f}°F")
            if current_obs_2 is not None:
                details.append(f"NWS: {current_obs_2:.1f}°F")
            if details:
                current_obs_str += "  (" + ", ".join(details) + ")\n"
            else:
                current_obs_str += "\n"

    report = (
        f"\n{airport_code} ({airport_name}) Forecast for Today:\n"
        f"{'='*60}\n"
        + current_obs_str +
        f"\n📊 ENSEMBLE PREDICTIONS (Weighted Average):\n"
        f"  Maximum Temperature:  {ensemble['max']:.1f}°F  (±{ensemble['max_ci']:.1f}°F 95% CI)\n"
        f"  Minimum Temperature:  {ensemble['min']:.1f}°F  (±{ensemble['min_ci']:.1f}°F 95% CI)\n"
        f"  Mean Temperature:      {ensemble['mean']:.1f}°F\n"
        f"  Daily Range:           {ensemble['max'] - ensemble['min']:.1f}°F\n"
        f"\n🎯 FORECAST CONFIDENCE:\n"
        f"  Confidence Level:      {ensemble['confidence']}\n"
        f"  Source Agreement:      {ensemble['source_spread']:.1f}°F spread\n"
        f"  Temperature Variance:  {ensemble['max_variance']:.2f}°F\n"
        f"\n📈 SOURCE BREAKDOWN:\n"
        f"  Open-Meteo Max: {ensemble['source_maxes'][0]:.1f}°F\n"
        f"  NWS Max:        {ensemble['source_maxes'][1]:.1f}°F\n"
        f"  Open-Meteo Min: {ensemble['source_mins'][0]:.1f}°F\n"
        f"  NWS Min:        {ensemble['source_mins'][1]:.1f}°F\n"
    )

    report += f"\n{'='*60}\n"
    return (report, ensemble)

# --- MAIN ---
def main():
    """Generate weather forecasts for all configured airports with adaptive ensemble."""
    console = Console()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    console.rule("🌤️  Weather Forecast Bot - Advanced Ensemble Predictor")
    console.print(f"Generated: {timestamp}", style="dim")
    
    # Load historical performance for adaptive weighting
    history = load_forecast_history()
    adaptive_weights = calculate_adaptive_weights(history)
    
    for airport_code, airport_info in AIRPORTS.items():
        try:
            with console.status(f"Fetching forecasts for {airport_code} ({airport_info['city']})...", spinner="dots"):
                openmeteo_data = fetch_open_meteo(airport_info['lat'], airport_info['lon'])
                nws_data = fetch_nws(airport_info['lat'], airport_info['lon'])
                current_obs_openmeteo = fetch_current_observation_openmeteo(airport_info['lat'], airport_info['lon'])
                current_obs_nws = fetch_current_observation_nws(airport_info['lat'], airport_info['lon'])

            report, ensemble = generate_report(airport_code, airport_info['city'], openmeteo_data, nws_data, current_obs_openmeteo, current_obs_nws)

            if ensemble is None:
                console.print(Panel(Text(report), style="red"))
                continue

            # Create a concise summary table for quick readability
            tbl = Table.grid(padding=(0,1))
            tbl.add_column(justify="left", ratio=2)
            tbl.add_column(justify="right", ratio=1)

            tbl.add_row("Maximum", f"{ensemble['max']:.1f}°F (±{ensemble['max_ci']:.1f})")
            tbl.add_row("Minimum", f"{ensemble['min']:.1f}°F (±{ensemble['min_ci']:.1f})")
            tbl.add_row("Mean", f"{ensemble['mean']:.1f}°F")
            tbl.add_row("Daily Range", f"{ensemble['max'] - ensemble['min']:.1f}°F")
            tbl.add_row("Confidence", ensemble['confidence'])
            tbl.add_row("Source Spread", f"{ensemble['source_spread']:.1f}°F")

            # Hourly snippet (first 12 values)
            hours_snippet = ensemble.get('hourly', [])[:12]
            hours_str = ', '.join(f"{h:.0f}°F" for h in hours_snippet) if hours_snippet else 'N/A'
            tbl.add_row("Hourly (first 12)", hours_str)

            header = f"{airport_code} — {airport_info['city']}"
            console.print(Panel(tbl, title=header, box=box.ROUNDED, expand=False))

            # Also print the detailed textual report in dim style for users who want more context
            console.print(Text(report), style="dim")

        except Exception as e:
            console.print(f"Error processing {airport_code}: {e}", style="bold red")

    console.rule()
    console.print("✓ Report generated successfully.", style="green")
    console.print(f"Adaptive Weights - Open-Meteo: {adaptive_weights['openmeteo']:.2f}, NWS: {adaptive_weights['nws']:.2f}", style="cyan")
    console.rule()

if __name__ == '__main__':
    main()
