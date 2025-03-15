import requests
import json

# Replace with your SerpAPI key
API_KEY = "0d2faf6139bf7b9966d129825f3d1953cb98dd38f17c0c55148c5a88362a8765"

def get_current_location():
    """Get the current location (latitude and longitude) using an IP-based geolocation service."""
    try:
        response = requests.get("https://ipinfo.io")
        data = response.json()
        location = data["loc"].split(",")
        print(float(location[0]), float(location[1]))
        return float(location[0]), float(location[1])  # lat, lon
    except Exception as e:
        print(f"Error getting current location: {e}")
        return None

def search_nearby_places(api_key, lat, lon, query, radius=20000):
    """Search for nearby places using SerpAPI."""
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_maps",
        "q": query,
        "ll": f"@{lat},{lon},15.1z",
        "radius": radius,  # Radius in meters (20 km = 20000 meters)
        "type": "search",
        "api_key": api_key,
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except Exception as e:
        print(f"Error fetching data from SerpAPI: {e}")
        return None

def main():
    # Get current location
    current_location = get_current_location()
    if not current_location:
        print("Could not determine current location.")
        return

    lat, lon = current_location
    queries = ["doctor", "clinic", "hospital"]
    results = []

    # Search for each type of place
    for query in queries:
        print(f"Searching for {query}s...")
        data = search_nearby_places(API_KEY, lat, lon, query)
        if data and "local_results" in data:
            for place in data["local_results"]:
                results.append({
                    "name": place.get("title", "N/A"),
                    "type": query,
                    "address": place.get("address", "N/A"),
                    "latitude": place.get("gps_coordinates", {}).get("latitude"),
                    "longitude": place.get("gps_coordinates", {}).get("longitude"),
                })

    # Save results to a JSON file
    with open("nearby_places.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Data saved to nearby_places.json. Found {len(results)} places.")

if __name__ == "__main__":
    main()  