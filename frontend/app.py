# Streamlit chat UI
import requests
import streamlit as st

API_URL = "http://localhost:8000"
IP_API_URL = "http://ip-api.com/json/"

st.set_page_config(
    page_title="Customer Experience AI",
    page_icon="☕",
    layout="centered",
    initial_sidebar_state="expanded",
)


def get_location_from_ip():
    """Fetch location from IP geolocation API."""
    try:
        response = requests.get(IP_API_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return {
                    "lat": data.get("lat", 40.7128),
                    "lng": data.get("lon", -74.0060),
                    "city": data.get("city", "Unknown"),
                    "country": data.get("country", "Unknown"),
                    "detected": True
                }
    except Exception:
        pass
    return {
        "lat": 40.7128,
        "lng": -74.0060,
        "city": "New York",
        "country": "USA",
        "detected": False
    }


def login(username: str, password: str):
    """Authenticate user with backend."""
    try:
        response = requests.post(
            f"{API_URL}/login",
            json={"username": username, "password": password},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {"success": False, "message": "Connection error"}


# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "location" not in st.session_state:
    st.session_state.location = get_location_from_ip()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "use_mock_data" not in st.session_state:
    st.session_state.use_mock_data = True


# Login page
if not st.session_state.authenticated:
    st.title("☕ Customer Experience AI")
    st.caption("Please login to continue")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username and password:
                result = login(username, password)
                if result.get("success"):
                    st.session_state.authenticated = True
                    st.session_state.user_id = result["user_id"]
                    st.session_state.user_name = result["name"]
                    st.rerun()
                else:
                    st.error(result.get("message", "Login failed"))
            else:
                st.warning("Please enter username and password")
    
    st.divider()
    st.caption("Demo credentials:")
    st.code("john / john123\nsarah / sarah123")

else:
    # Main chat interface
    st.title("☕ Customer Experience AI")
    st.caption(f"Welcome back, {st.session_state.user_name}!")
    
    # Sidebar
    with st.sidebar:
        st.header(f"👤 {st.session_state.user_name}")
        
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.user_name = None
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        st.header("📍 Location")
        loc = st.session_state.location
        if loc["detected"]:
            st.success(f"Auto-detected: {loc['city']}, {loc['country']}")
        else:
            st.warning("Using default location")
        
        st.caption(f"Coordinates: ({loc['lat']:.4f}, {loc['lng']:.4f})")
        
        with st.expander("🔧 Override Location (Demo)"):
            st.info(
                "**For Demo Purposes Only**\n\n"
                "In production, location is auto-detected via IP geolocation. "
                "For precise location, a real app would use browser Geolocation API."
            )
            
            new_lat = st.number_input("Latitude", value=loc["lat"], format="%.4f", key="lat")
            new_lng = st.number_input("Longitude", value=loc["lng"], format="%.4f", key="lng")
            
            if st.button("Update Location"):
                st.session_state.location = {
                    "lat": new_lat, "lng": new_lng,
                    "city": "Custom", "country": "Override", "detected": False
                }
                st.rerun()
        
        st.header("🏪 Store Data Source")
        use_mock = st.radio(
            "Where should recommendations come from?",
            options=["Live data (OpenStreetMap)", "Mock data (demo)"],
            index=1 if st.session_state.use_mock_data else 0,
            help="Live data pulls real nearby venues via OpenStreetMap. Mock data uses the bundled sample stores.",
        )
        st.session_state.use_mock_data = use_mock != "Live data (OpenStreetMap)"
        if st.session_state.use_mock_data:
            st.info("Using bundled demo stores & promotions.")
        else:
            st.success("Using live OpenStreetMap venues (beta).")

        st.divider()
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Get current location
    lat = st.session_state.location["lat"]
    lng = st.session_state.location["lng"]
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("How can I help you today?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Send to API with conversation history
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "user_id": st.session_state.user_id,
                            "message": prompt,
                            "lat": lat,
                            "lng": lng,
                            "use_mock_data": st.session_state.use_mock_data,
                            "conversation_history": st.session_state.messages[:-1]
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        bot_response = response.json()["response"]
                    else:
                        bot_response = f"Error: {response.status_code}"
                except requests.exceptions.ConnectionError:
                    bot_response = "⚠️ Cannot connect to backend. Make sure the API is running."
                except Exception as e:
                    bot_response = f"Error: {str(e)}"
            
            st.markdown(bot_response)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
