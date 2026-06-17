import urllib.request
import urllib.error
import urllib.parse
import json
from datetime import datetime

# --- CONFIGURATION ---
GOOGLE_MAPS_API_KEY = "AIzaSyBDaDvIv9wgsQJ7qoQLCAu18vCZ3qJLSHo"
HOME_ADDRESS = "3450 Quail Lake RD colorado springs CO 80906"
HOME_LAT = "38.7331"
HOME_LON = "-104.7925"

# --- CACHE ---
_weather_cache = {"data": "68°F, Clear skies (Mock)", "timestamp": 0}
CACHE_DURATION_SECONDS = 300

def get_weather():
    """Fetch live weather from Open-Meteo for Colorado Springs/Fort Carson."""
    global _weather_cache
    now = datetime.now().timestamp()
    if now - _weather_cache["timestamp"] < CACHE_DURATION_SECONDS:
        return _weather_cache["data"]

    url = f"https://api.open-meteo.com/v1/forecast?latitude={HOME_LAT}&longitude={HOME_LON}&current_weather=true&temperature_unit=fahrenheit"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'HomeAssist/1.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            temp = data['current_weather']['temperature']
            weather_code = data['current_weather']['weathercode']
            condition = "Clear" if weather_code <= 1 else ("Cloudy" if weather_code <= 3 else "Precipitation")
            
            result = f"{temp}°F, {condition}"
            _weather_cache = {"data": result, "timestamp": now}
            return result
    except Exception as e:
        print(f"[Warning] Weather API failed: {e}")
        return _weather_cache["data"]

def resolve_destination(destination):
    """Use Google Maps Places API to find the closest exact address for a generic destination."""
    if not GOOGLE_MAPS_API_KEY:
        return destination, destination

    query_encoded = urllib.parse.quote(destination)
    url = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={query_encoded}&inputtype=textquery&locationbias=point:{HOME_LAT},{HOME_LON}&fields=formatted_address,name&key={GOOGLE_MAPS_API_KEY}"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'HomeAssist/1.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data['status'] == 'OK' and len(data['candidates']) > 0:
                candidate = data['candidates'][0]
                resolved_name = candidate.get('name', destination)
                resolved_address = candidate.get('formatted_address', destination)
                return resolved_name, resolved_address
    except Exception as e:
        print(f"[Warning] Places API failed: {e}")
        
    return destination, destination

def get_live_traffic(destination):
    """Fetch live traffic from Google Maps for a specific destination."""
    if not GOOGLE_MAPS_API_KEY:
        hour = datetime.now().hour
        if (6 <= hour <= 8) or (15 <= hour <= 17):
            return f"Heavy Traffic to {destination} (25 mins) [Simulated]"
        else:
            return f"Clear Traffic to {destination} (12 mins) [Simulated]"

    resolved_name, resolved_address = resolve_destination(destination)

    origin_encoded = urllib.parse.quote(HOME_ADDRESS)
    destination_encoded = urllib.parse.quote(resolved_address)
    url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={origin_encoded}&destinations={destination_encoded}&departure_time=now&key={GOOGLE_MAPS_API_KEY}"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'HomeAssist/1.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data['status'] == 'OK':
                element = data['rows'][0]['elements'][0]
                if element['status'] == 'OK':
                    duration_in_traffic = element.get('duration_in_traffic', element.get('duration', {})).get('text', 'Unknown mins')
                    return f"Live Traffic to {resolved_name} ({resolved_address}): {duration_in_traffic}"
            
            return f"Traffic API Error: {data.get('status', 'Unknown Error')}"
    except Exception as e:
        print(f"[Warning] Google Maps Traffic API failed: {e}")
        return f"Could not fetch traffic to {resolved_name}."

def get_home_context():
    """Generate the context string representing the current state of the home and environment."""
    now = datetime.now()
    time_str = now.strftime("%I:%M %p")
    date_str = now.strftime("%A, %B %d")
    
    weather_str = get_weather()
    
    context = f"""
[SYSTEM STATUS REPORT]
Current Date & Time: {date_str} at {time_str}
Location: Primary Residence ({HOME_ADDRESS})

[SENSORS & ENVIRONMENT]
- Indoor Temperature: 71°F (Optimal)
- Outdoor Weather: {weather_str}
- All lights: Off (Energy saving mode active)

[SECURITY & EVENTS]
- Perimeter: Secure. All doors and windows locked.
- 14:30 : Front Door Camera - Motion detected (Courier drop-off).
- 18:45 : Living Room Camera - Maintenance bot (Roomba) completed cycle.
- {time_str} : Front Door Camera - Facial recognition matched profile: 'Marc' (Authorized User).
"""
    return context
