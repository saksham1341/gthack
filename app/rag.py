# RAG pipeline with ChromaDB
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

DATA_DIR = Path(__file__).parent.parent / "data"

# Initialize ChromaDB client (in-memory for simplicity)
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

# Collections
store_collection = None
memory_collection = None
profile_collection = None


def initialize_vectorstore():
    """Initialize ChromaDB with store and promotion data."""
    global store_collection, memory_collection, profile_collection
    
    # Create store info collection
    store_collection = chroma_client.get_or_create_collection(
        name="store_info",
        metadata={"description": "Store and promotion information"}
    )
    
    # Create user memory collection
    memory_collection = chroma_client.get_or_create_collection(
        name="user_memory",
        metadata={"description": "Long-term user conversation memory"}
    )
    
    # Create user profile collection (preferences, purchase history)
    profile_collection = chroma_client.get_or_create_collection(
        name="user_profile",
        metadata={"description": "User preferences and purchase history"}
    )
    
    # Check if store collection already populated
    if store_collection.count() > 0:
        return
    
    # Load store data
    with open(DATA_DIR / "stores.json", "r") as f:
        stores = json.load(f)
    with open(DATA_DIR / "promotions.json", "r") as f:
        promotions = json.load(f)
    
    documents = []
    metadatas = []
    ids = []
    
    # Add store info
    for store in stores:
        doc = f"Store: {store['name']}. Type: {store['type']}. "
        doc += f"Hours: {store['hours']['open']} to {store['hours']['close']}. "
        doc += f"Popular items: {', '.join(store['popular_items'])}."
        
        documents.append(doc)
        metadatas.append({"type": "store", "store_id": store["store_id"]})
        ids.append(f"store_{store['store_id']}")
    
    # Add promotion info
    for promo in promotions:
        doc = f"Promotion: {promo['title']}. {promo['description']}. "
        doc += f"Applies to: {', '.join(promo['applicable_items'])}."
        
        documents.append(doc)
        metadatas.append({"type": "promotion", "store_id": promo["store_id"], "promo_id": promo["promo_id"]})
        ids.append(f"promo_{promo['promo_id']}")
    
    # Add to collection
    store_collection.add(documents=documents, metadatas=metadatas, ids=ids)
    
    # Initialize user profiles from users.json (seed data)
    _seed_user_profiles()


def _seed_user_profiles():
    """Seed user profiles from users.json if not already present."""
    global profile_collection
    
    with open(DATA_DIR / "users.json", "r") as f:
        users = json.load(f)
    
    for user in users:
        user_id = user["user_id"]
        
        # Check if profile already exists
        existing = profile_collection.get(ids=[f"profile_{user_id}"])
        if existing["ids"]:
            continue
        
        # Create profile document
        profile_data = {
            "preferences": user.get("preferences", {}),
            "purchase_history": user.get("purchase_history", []),
            "loyalty_points": user.get("loyalty_points", 0)
        }
        
        doc = _profile_to_doc(profile_data)
        
        profile_collection.add(
            documents=[doc],
            metadatas=[{
                "user_id": user_id,
                "profile_json": json.dumps(profile_data),
                "updated_at": datetime.now().isoformat()
            }],
            ids=[f"profile_{user_id}"]
        )


def _profile_to_doc(profile_data: Dict) -> str:
    """Convert profile data to searchable document string."""
    parts = []
    prefs = profile_data.get("preferences", {})
    
    if prefs.get("favorite_drinks"):
        parts.append(f"Favorite drinks: {', '.join(prefs['favorite_drinks'])}")
    if prefs.get("dietary"):
        parts.append(f"Dietary preferences: {', '.join(prefs['dietary'])}")
    if prefs.get("preferred_temperature"):
        parts.append(f"Temperature preference: {prefs['preferred_temperature']}")
    
    history = profile_data.get("purchase_history", [])
    if history:
        items = [h["item"] for h in history[-5:]]
        parts.append(f"Recent purchases: {', '.join(items)}")
    
    if profile_data.get("loyalty_points"):
        parts.append(f"Loyalty points: {profile_data['loyalty_points']}")
    
    return ". ".join(parts) if parts else "No preferences set"


def retrieve_relevant_info(query: str, n_results: int = 3) -> List[str]:
    """Retrieve relevant store/promotion info based on query."""
    global store_collection
    
    if store_collection is None:
        initialize_vectorstore()
    
    results = store_collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0] if results["documents"] else []


# ============ User Profile Functions ============

def get_user_profile_data(user_id: str) -> Optional[Dict]:
    """Get user's dynamic profile data (preferences, history) from ChromaDB."""
    global profile_collection
    
    if profile_collection is None:
        initialize_vectorstore()
    
    result = profile_collection.get(ids=[f"profile_{user_id}"])
    
    if result["ids"] and result["metadatas"]:
        profile_json = result["metadatas"][0].get("profile_json", "{}")
        return json.loads(profile_json)
    
    return None


def update_user_profile(user_id: str, updates: Dict):
    """Update user's profile with new preferences or data."""
    global profile_collection
    
    if profile_collection is None:
        initialize_vectorstore()
    
    current = get_user_profile_data(user_id)
    if not current:
        current = {"preferences": {}, "purchase_history": [], "loyalty_points": 0}
    
    prefs = current.get("preferences", {})
    
    if "add_favorite_drink" in updates:
        drinks = prefs.get("favorite_drinks", [])
        new_drink = updates["add_favorite_drink"]
        if new_drink not in drinks:
            drinks.append(new_drink)
        prefs["favorite_drinks"] = drinks
    
    if "add_dietary" in updates:
        dietary = prefs.get("dietary", [])
        new_dietary = updates["add_dietary"]
        if new_dietary not in dietary:
            dietary.append(new_dietary)
        prefs["dietary"] = dietary
    
    if "set_temperature" in updates:
        prefs["preferred_temperature"] = updates["set_temperature"]
    
    if "add_purchase" in updates:
        history = current.get("purchase_history", [])
        history.append({
            "item": updates["add_purchase"],
            "date": datetime.now().strftime("%Y-%m-%d")
        })
        current["purchase_history"] = history[-10:]
    
    if "add_loyalty_points" in updates:
        current["loyalty_points"] = current.get("loyalty_points", 0) + updates["add_loyalty_points"]
    
    current["preferences"] = prefs
    
    doc = _profile_to_doc(current)
    
    profile_collection.update(
        ids=[f"profile_{user_id}"],
        documents=[doc],
        metadatas=[{
            "user_id": user_id,
            "profile_json": json.dumps(current),
            "updated_at": datetime.now().isoformat()
        }]
    )


def add_learned_preference(user_id: str, preference_text: str):
    """Parse a learned preference from conversation and update profile."""
    text_lower = preference_text.lower()
    updates = {}
    
    # Parse temperature preferences
    if "hot" in text_lower and ("drink" in text_lower or "prefer" in text_lower or "like" in text_lower):
        updates["set_temperature"] = "hot"
    elif "cold" in text_lower and ("drink" in text_lower or "prefer" in text_lower or "like" in text_lower):
        updates["set_temperature"] = "cold"
    elif "iced" in text_lower:
        updates["set_temperature"] = "cold"
    
    # Parse dietary preferences
    dietary_keywords = ["vegan", "vegetarian", "gluten-free", "dairy-free", "healthy", "low-sugar", "keto"]
    for keyword in dietary_keywords:
        if keyword in text_lower:
            updates["add_dietary"] = keyword
            break
    
    # Parse drink preferences
    drink_keywords = ["coffee", "latte", "cappuccino", "espresso", "tea", "smoothie", "cocoa", 
                      "frappuccino", "matcha", "chai", "mocha", "americano"]
    for drink in drink_keywords:
        if drink in text_lower:
            updates["add_favorite_drink"] = drink
            break
    
    if updates:
        update_user_profile(user_id, updates)


# ============ Memory Functions ============

def save_memory(user_id: str, memory_type: str, content: str):
    """Save a memory entry for a user."""
    global memory_collection
    
    if memory_collection is None:
        initialize_vectorstore()
    
    timestamp = datetime.now().isoformat()
    memory_id = f"{user_id}_{memory_type}_{timestamp}"
    
    memory_collection.add(
        documents=[content],
        metadatas=[{"user_id": user_id, "memory_type": memory_type, "timestamp": timestamp}],
        ids=[memory_id]
    )


def retrieve_memories(user_id: str, query: str, n_results: int = 5) -> List[str]:
    """Retrieve relevant memories for a user based on query."""
    global memory_collection
    
    if memory_collection is None:
        initialize_vectorstore()
    
    results = memory_collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"user_id": user_id}
    )
    return results["documents"][0] if results["documents"] else []
