import requests
from datetime import datetime
from statistics import median, mean

# --- CONFIGURATION ---
AIRPORTS = {
    'KAUS': {'lat': 30.1944, 'lon': -97.67, 'city': 'Austin'},
    'KDFW': {'lat': 32.8968, 'lon': -97.038, 'city': 'Dallas-Fort Worth'}
}

# --- BIAS CORRECTION AND WEIGHTED ENSEMBLE CONFIG ---
HISTORICAL_BIAS = {
    'openmeteo': 0.0,
    'nws': 0.0
}
SOURCE_WEIGHTS = {
    'openmeteo': 0.5,
    'nws': 0.5
}

# --- WEATHER SOURCES ---
def fetch_open_meteo(lat, lon):
    """Fetch hourly temperature forecast from Open-Meteo (free API, no auth required)."""
    try:
        url = f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m&forecast_days=1&timezone=auto&temperature_unit=fahrenheit'
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        temps = data['hourly']['temperature_2m']
        return temps
    except requests.RequestException as e:
        print(f"Error fetching from Open-Meteo: {e}")
        return []

def fetch_nws(lat, lon):
    """Fetch hourly temperature forecast from NOAA/NWS (US government API)."""
    try:
        headers = {'User-Agent': 'weather-bot/1.0'}
        points_url = f'https://api.weather.gov/points/{lat},{lon}'
        points_resp = requests.get(points_url, headers=headers, timeout=10)
        points_resp.raise_for_status()
        forecast_url = points_resp.json()['properties']['forecastHourly']
        forecast_resp = requests.get(forecast_url, headers=headers, timeout=10)
        forecast_resp.raise_for_status()
        periods = forecast_resp.json()['properties']['periods']
        temps = [p['temperature'] for p in periods if p['isDaytime']][:24]
        return temps
    except requests.RequestException as e:
        print(f"Error fetching from NWS: {e}")
        return []

# --- ENSEMBLE LOGIC ---

def robust_ensemble_max(temps_list):
    """
    Align hourly temps from multiple sources, extract daily max/min, apply bias correction,
    and ensemble using weighted mean. Returns ensemble max, min, mean, confidence, and details.
    """
    if not temps_list or any(not t for t in temps_list):
        return None, 'UNAVAILABLE', []
    
    # Truncate all to shortest length to align hourly data
    min_len = min(len(t) for t in temps_list)
    aligned = [t[:min_len] for t in temps_list]
    
    # Extract max and min for each source
    maxes = [max(t) for t in aligned]
    mins = [min(t) for t in aligned]
    
    # Apply bias correction
    corrected_maxes = [maxes[0] + HISTORICAL_BIAS['openmeteo'], maxes[1] + HISTORICAL_BIAS['nws']]
    corrected_mins = [mins[0] + HISTORICAL_BIAS['openmeteo'], mins[1] + HISTORICAL_BIAS['nws']]
    
    # Weighted ensemble for max and min
    weights = [SOURCE_WEIGHTS['openmeteo'], SOURCE_WEIGHTS['nws']]
    ensemble_max = sum(w * m for w, m in zip(weights, corrected_maxes))
    ensemble_min = sum(w * m for w, m in zip(weights, corrected_mins))
    
    # Confidence: HIGH if max spread <= 3°F, LOW if > 3°F
    spread = max(corrected_maxes) - min(corrected_maxes)
    confidence = 'HIGH' if spread <= 3 else 'LOW'
    
    return ensemble_max, ensemble_min, confidence, corrected_maxes, corrected_mins, spread

# --- REPORT GENERATION ---
def generate_report(airport_code, airport_name, temps_openmeteo, temps_nws):
    """Generate a comprehensive forecast report with ensemble predictions."""
    ensemble_max, ensemble_min, confidence, corrected_maxes, corrected_mins, spread = robust_ensemble_max([temps_openmeteo, temps_nws])
    
    if confidence == 'UNAVAILABLE':
        return f"{airport_code} ({airport_name}): Data unavailable. Check API connections.\n"
    
    # Show both raw and corrected maxes/mins for transparency
    raw_maxes = [max(temps_openmeteo), max(temps_nws)]
    raw_mins = [min(temps_openmeteo), min(temps_nws)]
    
    report = (
        f"\n{airport_code} ({airport_name}) Forecast for Today:\n"
        f"{'='*50}\n"
        f"Open-Meteo Max (raw): {raw_maxes[0]:.1f}°F → (corrected): {corrected_maxes[0]:.1f}°F\n"
        f"NWS Max (raw): {raw_maxes[1]:.1f}°F → (corrected): {corrected_maxes[1]:.1f}°F\n"
        f"\n**Ensemble Max: {ensemble_max:.1f}°F**\n"
        f"**Ensemble Min: {ensemble_min:.1f}°F**\n"
        f"Forecast Range: {ensemble_max - ensemble_min:.1f}°F\n"
        f"Confidence: {confidence} (source disagreement: {spread:.1f}°F)\n"
    )
    return report

# --- MAIN ---
def main():
    """Generate weather forecasts for all configured airports."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*50}")
    print(f"Weather Forecast Bot - Robinhoon Betting Helper")
    print(f"Generated: {timestamp}")
    print(f"{'='*50}")
    
    full_report = ""
    for airport_code, airport_info in AIRPORTS.items():
        try:
            # Fetch forecasts from both sources
            openmeteo_temps = fetch_open_meteo(airport_info['lat'], airport_info['lon'])
            nws_temps = fetch_nws(airport_info['lat'], airport_info['lon'])
            
            # Generate report
            report = generate_report(airport_code, airport_info['city'], openmeteo_temps, nws_temps)
            full_report += report
        except Exception as e:
            print(f"Error processing {airport_code}: {e}")
    
    print(full_report)
    print(f"{'='*50}")
    print("Report generated successfully.")

if __name__ == '__main__':
    main()
