# H-002 | Customer Experience Automation
**Track:** Customer Experience & Conversational AI

## Problem Summary
Build a hyper-personalized customer support agent that:
- Knows customer history & context
- Provides location-aware recommendations
- Masks sensitive data before LLM processing

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                 │
│        "I'm cold" + Location (lat/lng) identified by backend            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         1. DATA MASKING LAYER                           │
│   • Regex-based PII detection (phone, email, SSN)                       │
│   • Replace with tokens: [PHONE_1], [EMAIL_1]                           │
│   • Store mapping for response de-masking                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      2. CONTEXT ENRICHMENT                              │
│   • Fetch user profile from local DB (preferences, history)             │
│   • Query nearby stores via location (mock/static data)                 │
│   • Retrieve active promotions/coupons                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     3. RAG PIPELINE                                     │
│   • Vector store (ChromaDB) with store info PDFs                        │
│   • Embed query → Retrieve relevant chunks                              │
│   • Inject into prompt context                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      4. LLM ORCHESTRATION                               │
│   • System prompt with persona + context                                │
│   • User message (masked) + enriched context                            │
│   • Generate personalized response                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       5. RESPONSE + UNMASK                              │
│   • De-mask any tokens in response                                      │
│   • Return to user via chat interface                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | FastAPI (Python) |
| **LLM** | Gemini API |
| **Vector DB** | ChromaDB |
| **Frontend** | Streamlit |
| **Data Masking** | Regex |
| **Mock Data** | JSON files |

## Project Structure

```
gthack/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── chat.py              # Chat endpoint & LLM logic
│   ├── masking.py           # PII detection & masking
│   ├── context.py           # User profile & location enrichment
│   └── rag.py               # RAG pipeline
├── data/
│   ├── users.json           # Mock user profiles
│   ├── stores.json          # Mock store data with locations
│   └── promotions.json      # Active coupons/deals
├── frontend/
│   └── app.py               # Streamlit chat UI
├── requirements.txt
├── env.example
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp env.example .env

# Run backend
uvicorn app.main:app --reload --port 8000

# Run frontend (new terminal)
streamlit run frontend/app.py
```
