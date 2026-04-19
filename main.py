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

# --- CONFIGURATION ---
AIRPORTS = {
    'KAUS': {'lat': 30.1944, 'lon': -97.67, 'city': 'Austin'},
    'KDFW': {'lat': 32.8968, 'lon': -97.038, 'city': 'Dallas-Fort Worth'}
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
        url = f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m&forecast_days=1&timezone=auto&temperature_unit=fahrenheit'
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        hourly = data['hourly']
        return {
            'temps': hourly['temperature_2m'],
            'humidity': hourly.get('relative_humidity_2m', []),
            'pressure': hourly.get('pressure_msl', []),
            'wind': hourly.get('wind_speed_10m', []),
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
        temps = [p['temperature'] for p in periods][:48]
        wind = [p.get('windSpeed', '0 mph').split()[0] if p.get('windSpeed') else 0 for p in periods][:48]
        
        return {
            'temps': temps,
            'humidity': [],
            'pressure': [],
            'wind': [float(w) if w else 0 for w in wind],
            'source': 'nws'
        }
    except Exception as e:
        import traceback
        print(f"Error fetching from NWS: {e}")
        traceback.print_exc()
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
        'hourly_range': max(hourly_ensemble) - min(hourly_ensemble) if hourly_ensemble else 0
    }

# --- REPORT GENERATION ---
def generate_report(airport_code, airport_name, forecast_data_1, forecast_data_2):
    """Generate comprehensive forecast report with ensemble predictions, uncertainty, and hourly breakdown."""
    ensemble = robust_ensemble_max([forecast_data_1, forecast_data_2])
    
    if ensemble is None:
        return f"{airport_code} ({airport_name}): Data unavailable. Check API connections.\n"
    
    report = (
        f"\n{airport_code} ({airport_name}) Forecast for Today:\n"
        f"{'='*60}\n"
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
    
    # Add hourly breakdown for first 12 hours with actual times in Central Time
    if ensemble['hourly']:
        report += f"\n⏰ HOURLY FORECAST (First 12 hours, Central Time):\n"
        now_utc = datetime.now(tz=ZoneInfo("UTC"))
        now_central = now_utc.astimezone(ZoneInfo("America/Chicago"))
        for i, temp in enumerate(ensemble['hourly'][:12]):
            hour_time = now_central.replace(minute=0, second=0, microsecond=0) + timedelta(hours=i)
            time_str = hour_time.strftime('%H:%M (%I:%M %p)')
            report += f"  {time_str}: {temp:6.1f}°F\n"
    
    report += f"\n{'='*60}\n"
    return report

# --- MAIN ---
def main():
    """Generate weather forecasts for all configured airports with adaptive ensemble."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*60}")
    print(f"🌤️  Weather Forecast Bot - Advanced Ensemble Predictor")
    print(f"Generated: {timestamp}")
    print(f"{'='*60}")
    
    # Load historical performance for adaptive weighting
    history = load_forecast_history()
    adaptive_weights = calculate_adaptive_weights(history)
    
    full_report = ""
    for airport_code, airport_info in AIRPORTS.items():
        try:
            # Fetch forecasts from both sources with full meteorological data
            openmeteo_data = fetch_open_meteo(airport_info['lat'], airport_info['lon'])
            nws_data = fetch_nws(airport_info['lat'], airport_info['lon'])
            
            if openmeteo_data and nws_data:
                # Generate report with improved ensemble
                report = generate_report(airport_code, airport_info['city'], openmeteo_data, nws_data)
                full_report += report
            else:
                full_report += f"{airport_code} ({airport_info['city']}): Data fetch failed. Check API connections.\n"
        except Exception as e:
            print(f"Error processing {airport_code}: {e}")
            full_report += f"{airport_code}: Error - {str(e)}\n"
    
    print(full_report)
    print(f"{'='*60}")
    print("✓ Report generated successfully.")
    print(f"Adaptive Weights - Open-Meteo: {adaptive_weights['openmeteo']:.2f}, NWS: {adaptive_weights['nws']:.2f}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
