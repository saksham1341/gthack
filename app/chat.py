# Chat endpoint & LLM logic with LangGraph
import os
from typing import Dict, List, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from app.context import enrich_context
from app.masking import mask_pii, unmask_pii
from app.rag import initialize_vectorstore, retrieve_relevant_info, save_memory, retrieve_memories, add_learned_preference

load_dotenv()

# LangSmith configuration (auto-enabled when env vars are set)
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_key
# LANGCHAIN_PROJECT=customer-experience-ai

# Initialize RAG vectorstore on import
initialize_vectorstore()


class ChatState(TypedDict):
    """State for the chat workflow."""
    user_id: str
    user_message: str
    lat: float
    lng: float
    use_mock_data: bool
    conversation_history: List[Dict]
    enriched_context: Dict
    long_term_memories: List[str]
    rag_results: list
    masked_message: str
    masked_context: str
    pii_mapping: Dict
    llm_response: str
    final_response: str


def context_enrichment_node(state: ChatState) -> ChatState:
    """Node 1: Enrich context with user profile, nearby stores, and long-term memory."""
    context = enrich_context(
        user_id=state["user_id"],
        lat=state["lat"],
        lng=state["lng"],
        use_mock_data=state.get("use_mock_data", True)
    )
    
    # Retrieve relevant long-term memories
    memories = retrieve_memories(
        user_id=state["user_id"],
        query=state["user_message"],
        n_results=3
    )
    
    return {"enriched_context": context, "long_term_memories": memories}


def rag_retrieval_node(state: ChatState) -> ChatState:
    """Node 2: Retrieve relevant info from vector store."""
    query = state["user_message"]
    rag_results = retrieve_relevant_info(query, n_results=3)
    return {"rag_results": rag_results}


def pii_masking_node(state: ChatState) -> ChatState:
    """Node 3: Mask PII in message and context before LLM."""
    # Mask user message
    masked_message, mapping = mask_pii(state["user_message"])
    
    # Build context string and mask it
    context = state["enriched_context"]
    context_str = _build_context_string(
        context, 
        state["rag_results"], 
        state["long_term_memories"],
        state["conversation_history"]
    )
    masked_context, mapping = mask_pii(context_str, existing_mapping=mapping)
    
    return {
        "masked_message": masked_message,
        "masked_context": masked_context,
        "pii_mapping": mapping
    }


def _build_context_string(context: Dict, rag_results: list, memories: list, history: list) -> str:
    """Build a context string for the LLM prompt."""
    parts = []
    
    # User info
    if context.get("user"):
        user = context["user"]
        parts.append(f"Customer: {user['name']}")
        if user.get("preferences"):
            prefs = user["preferences"]
            if prefs.get("favorite_drinks"):
                parts.append(f"Favorite drinks: {', '.join(prefs['favorite_drinks'])}")
            if prefs.get("dietary"):
                parts.append(f"Dietary preferences: {', '.join(prefs['dietary'])}")
            if prefs.get("preferred_temperature"):
                parts.append(f"Prefers: {prefs['preferred_temperature']} drinks")
        if user.get("loyalty_points"):
            parts.append(f"Loyalty points: {user['loyalty_points']}")
    
    # Long-term memories
    if memories:
        parts.append(f"Previous interactions: {'; '.join(memories)}")
    
    # Nearby stores
    if context.get("nearby_stores"):
        stores_info = []
        for store in context["nearby_stores"][:3]:
            stores_info.append(f"{store['name']} ({store['distance_m']}m away, open {store['hours']['open']}-{store['hours']['close']})")
        parts.append(f"Nearby stores: {'; '.join(stores_info)}")
    
    # Promotions
    if context.get("promotions"):
        promo_info = [p["title"] for p in context["promotions"][:3]]
        parts.append(f"Available promotions: {', '.join(promo_info)}")
    
    # RAG results
    if rag_results:
        parts.append(f"Additional info: {' '.join(rag_results)}")
    
    # Recent conversation history
    if history:
        recent = history[-6:]  # Last 3 exchanges
        history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in recent])
        parts.append(f"Recent conversation:\n{history_str}")
    
    return "\n".join(parts)


def llm_generation_node(state: ChatState) -> ChatState:
    """Node 4: Generate response using Gemini."""
    import google.generativeai as genai
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"llm_response": "Error: GOOGLE_API_KEY not configured"}
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = f"""You are a hyper-local concierge that recommends nearby places and products.
Your job is to tell the customer what they should do next (e.g., "Stop by X and grab Y").
Always reference the most relevant nearby store or action from the context.
Never claim that you can make, prepare, or serve items yourself—only recommend external locations.
Keep responses concise (2-3 sentences) and grounded in the provided context and history.

CUSTOMER CONTEXT & HISTORY:
{state["masked_context"]}

CUSTOMER MESSAGE: {state["masked_message"]}

Respond helpfully:"""
    
    response = model.generate_content(prompt)
    return {"llm_response": response.text}


def unmask_response_node(state: ChatState) -> ChatState:
    """Node 5: Unmask PII in the response."""
    final_response = unmask_pii(state["llm_response"], state["pii_mapping"])
    return {"final_response": final_response}


def persona_update_node(state: ChatState) -> ChatState:
    """Node 6: Analyze conversation and update user persona/memory."""
    import google.generativeai as genai
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return state
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Analyze conversation for insights
    prompt = f"""Analyze this customer interaction and extract any NEW preferences or facts learned.
Only output if there's something genuinely new to learn. Be concise (1 sentence max).
If nothing new, output "NONE".

Customer message: {state["user_message"]}
Bot response: {state["final_response"]}
Current known preferences: {state["enriched_context"].get("user", {}).get("preferences", {})}

New insight (or NONE):"""
    
    try:
        response = model.generate_content(prompt)
        insight = response.text.strip()
        
        if insight and insight.upper() != "NONE":
            # Update user profile with learned preference (persists to ChromaDB)
            add_learned_preference(
                user_id=state["user_id"],
                preference_text=insight
            )
            
            # Save to long-term memory
            save_memory(
                user_id=state["user_id"],
                memory_type="learned_fact",
                content=insight
            )
            
            # Also save conversation summary
            summary = f"User asked about '{state['user_message'][:50]}...', recommended based on their preferences."
            save_memory(
                user_id=state["user_id"],
                memory_type="conversation_summary",
                content=summary
            )
    except Exception:
        pass  # Don't fail the whole flow if persona update fails
    
    return state


def build_chat_graph() -> StateGraph:
    """Build the LangGraph workflow."""
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("context_enrichment", context_enrichment_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("pii_masking", pii_masking_node)
    workflow.add_node("llm_generation", llm_generation_node)
    workflow.add_node("unmask_response", unmask_response_node)
    workflow.add_node("persona_update", persona_update_node)
    
    # Define edges
    workflow.set_entry_point("context_enrichment")
    workflow.add_edge("context_enrichment", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "pii_masking")
    workflow.add_edge("pii_masking", "llm_generation")
    workflow.add_edge("llm_generation", "unmask_response")
    workflow.add_edge("unmask_response", "persona_update")
    workflow.add_edge("persona_update", END)
    
    return workflow.compile()


# Compiled graph instance
chat_graph = build_chat_graph()


def process_message(
    user_id: str, 
    message: str, 
    lat: float, 
    lng: float,
    conversation_history: List[Dict] = None,
    use_mock_data: bool = True
) -> str:
    """
    Process a user message through the LangGraph workflow.
    
    Returns the final response string.
    """
    if conversation_history is None:
        conversation_history = []
    
    initial_state: ChatState = {
        "user_id": user_id,
        "user_message": message,
        "lat": lat,
        "lng": lng,
        "use_mock_data": use_mock_data,
        "conversation_history": conversation_history,
        "enriched_context": {},
        "long_term_memories": [],
        "rag_results": [],
        "masked_message": "",
        "masked_context": "",
        "pii_mapping": {},
        "llm_response": "",
        "final_response": ""
    }
    
    # LangSmith config for tracing
    config = {
        "run_name": f"chat_{user_id}",
        "metadata": {
            "user_id": user_id,
            "lat": lat,
            "lng": lng,
            "message_preview": message[:50] if len(message) > 50 else message,
            "data_source": "mock" if use_mock_data else "live"
        },
        "tags": ["customer-chat", user_id]
    }
    
    result = chat_graph.invoke(initial_state, config=config)
    return result["final_response"]
