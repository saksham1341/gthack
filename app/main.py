# FastAPI app entry point
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.chat import process_message
from app.context import load_json

app = FastAPI(
    title="Customer Experience AI",
    description="Hyper-personalized customer support agent",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    success: bool
    user_id: Optional[str] = None
    name: Optional[str] = None
    message: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    user_id: str
    message: str
    lat: float
    lng: float
    use_mock_data: bool = True
    conversation_history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    response: str


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "Customer Experience AI"}


@app.post("/login", response_model=LoginResponse)
def login(request: LoginRequest):
    """
    Authenticate user with username and password.
    """
    users = load_json("users.json")
    
    for user in users:
        if user.get("username") == request.username and user.get("password") == request.password:
            return LoginResponse(
                success=True,
                user_id=user["user_id"],
                name=user["name"],
                message="Login successful"
            )
    
    return LoginResponse(
        success=False,
        message="Invalid username or password"
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Process a chat message from the user.
    
    - Enriches context with user profile and nearby stores
    - Retrieves relevant info via RAG
    - Masks PII before LLM processing
    - Updates user persona based on conversation
    - Returns personalized response
    """
    # Convert conversation history to list of dicts
    history = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]
    
    response = process_message(
        user_id=request.user_id,
        message=request.message,
        lat=request.lat,
        lng=request.lng,
        conversation_history=history,
        use_mock_data=request.use_mock_data
    )
    return ChatResponse(response=response)
