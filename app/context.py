# User profile & location enrichment
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import requests

from app.rag import get_user_profile_data

DATA_DIR = Path(__file__).parent.parent / "data"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def load_json(filename: str) -> List[Dict]:
    """Load JSON data from file."""
    with open(DATA_DIR / filename, "r") as f:
        return json.load(f)


def get_user_profile(user_id: str) -> Optional[Dict]:
    """
    Fetch user profile by user_id.
    Combines static auth data from users.json with dynamic data from ChromaDB.
    """
    users = load_json("users.json")
    static_user = None
    
    for user in users:
        if user["user_id"] == user_id:
            static_user = user
            break
    
    if not static_user:
        return None
    
    # Get dynamic profile data from ChromaDB
    dynamic_data = get_user_profile_data(user_id)
    
    # Merge static and dynamic data
    profile = {
        "user_id": static_user["user_id"],
        "name": static_user["name"],
        "phone": static_user.get("phone"),
        "email": static_user.get("email"),
    }
    
    if dynamic_data:
        profile["preferences"] = dynamic_data.get("preferences", {})
        profile["purchase_history"] = dynamic_data.get("purchase_history", [])
        profile["loyalty_points"] = dynamic_data.get("loyalty_points", 0)
    else:
        # Fallback to static data if no dynamic data exists yet
        profile["preferences"] = static_user.get("preferences", {})
        profile["purchase_history"] = static_user.get("purchase_history", [])
        profile["loyalty_points"] = static_user.get("loyalty_points", 0)
    
    return profile


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance between two coordinates in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def get_mock_stores(lat: float, lng: float, radius_m: float = 500) -> List[Dict]:
    """Find stores within radius (meters) of given coordinates."""
    stores = load_json("stores.json")
    nearby = []
    
    for store in stores:
        store_lat = store["location"]["lat"]
        store_lng = store["location"]["lng"]
        distance = haversine_distance(lat, lng, store_lat, store_lng)
        
        if distance <= radius_m:
            store_with_distance = store.copy()
            store_with_distance["distance_m"] = round(distance)
            nearby.append(store_with_distance)
    
    # Sort by distance
    nearby.sort(key=lambda x: x["distance_m"])
    return nearby


def fetch_live_stores(lat: float, lng: float, radius_m: float = 800, limit: int = 6) -> List[Dict]:
    """Fetch nearby stores dynamically using OpenStreetMap Overpass API."""
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"~"cafe|restaurant|fast_food"](around:{radius_m},{lat},{lng});
      node["shop"~"coffee|convenience"](around:{radius_m},{lat},{lng});
    );
    out body;
    """
    try:
        response = requests.post(OVERPASS_URL, data={"data": query}, timeout=20)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    elements = data.get("elements", [])[:limit]
    stores = []
    for element in elements:
        tags = element.get("tags", {})
        name = tags.get("name") or tags.get("brand") or "Nearby Spot"
        store_lat = element.get("lat")
        store_lng = element.get("lon")
        if store_lat is None or store_lng is None:
            continue
        distance = haversine_distance(lat, lng, store_lat, store_lng)
        store_info = {
            "store_id": f"live_{element.get('id')}",
            "name": name,
            "type": tags.get("amenity") or tags.get("shop") or "venue",
            "location": {"lat": store_lat, "lng": store_lng},
            "hours": {"open": tags.get("opening_hours", "Unknown"), "close": ""},
            "distance_m": round(distance),
            "source": "live"
        }
        stores.append(store_info)

    stores.sort(key=lambda x: x["distance_m"])
    return stores


def get_store_promotions(store_id: str) -> List[Dict]:
    """Get active promotions for a store."""
    promotions = load_json("promotions.json")
    return [p for p in promotions if p["store_id"] == store_id]


def enrich_context(user_id: str, lat: float, lng: float, use_mock_data: bool = True) -> Dict:
    """
    Enrich context with user profile, nearby stores, and promotions.
    
    Returns dict with all enriched context for LLM.
    """
    context = {
        "user": None,
        "nearby_stores": [],
        "promotions": []
    }
    
    # Get user profile (merged static + dynamic from ChromaDB)
    user = get_user_profile(user_id)
    if user:
        context["user"] = user
    
    # Get nearby stores
    if use_mock_data:
        nearby_stores = get_mock_stores(lat, lng)
    else:
        nearby_stores = fetch_live_stores(lat, lng)
        # Fallback to mock if live data fails
        if not nearby_stores:
            nearby_stores = get_mock_stores(lat, lng)
    context["nearby_stores"] = nearby_stores
    
    # Get promotions for nearby stores (mock data only)
    if use_mock_data:
        for store in nearby_stores:
            store_promos = get_store_promotions(store["store_id"])
            context["promotions"].extend(store_promos)
    
    return context
