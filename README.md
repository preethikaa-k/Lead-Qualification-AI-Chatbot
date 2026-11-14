# Ashcombe & Rowe Estates — AI Lead Qualification Chatbot

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-30%2F30%20passing-brightgreen.svg)]()
[![Benchmark Pass Rate](https://img.shields.io/badge/benchmarks-100%25%20pass%20rate-success.svg)]()
[![Docker Ready](https://img.shields.io/badge/docker-ready-blue.svg)]()

Production-grade agentic lead qualification chatbot for **Ashcombe & Rowe Estates**, a fictional luxury UK estate agency specialising in prime residential properties (£1M+) across London, Surrey, and Oxford.

Built for the **Commversion AI & Data Team AI Engineer Assignment**.

---

## Architecture Overview

```mermaid
flowchart TD
    User([Visitor / Web Chat / CLI]) --> Router[Agent Orchestrator]
    
    subgraph Guardrails & Normalizers
        Router --> InjectionGuard[Prompt Injection & Jailbreak Defense]
        InjectionGuard --> InputParsers[Deterministic Regex Parsers\nBudget £1.5m / 950k, Postcode, Phone, Email]
    end

    subgraph Dual-Track Knowledge System
        InputParsers --> FAQCheck{FAQ / Company Query?}
        FAQCheck -->|Yes| FAQRAG[FAQ RAG Engine\nSemantic Vector Retrieval + 020 7946 0100 Fallback]
        FAQCheck -->|No| FlowGraph[Deterministic Conversation State Machine]
    end

    subgraph Flowchart Lanes
        FlowGraph -->|Start| S[Greet & Ask Name Once]
        S --> BS{Buy or Sell?}
        
        BS -->|Sell| LaneD[Lane D: Seller\npostcode in seller_coverage.csv?]
        BS -->|Buy| SpecProp{Specific Property?}
        
        SpecProp -->|Yes| LaneC[Lane C: Specific Property\nLook up name/address]
        SpecProp -->|No| CatChoice{New-Build or Resale?}
        
        CatChoice -->|New-Build| LaneA[Lane A: New-Build Buyer\nBudget >= 1M & Available at Postcode]
        CatChoice -->|Resale| LaneB[Lane B: Resale Buyer\nRecorded Budget & Available at Postcode]
    end

    subgraph Lead Qualification & Gatekeeper
        LaneA -->|Budget < 1M| M1[M1 Ending: Under 1M]
        LaneA -->|No NB at postcode| M2[M2 Ending: Postcode not catered]
        LaneA -->|None available <= budget| M3[M3 Ending: No matching available]
        LaneA -->|Qualified| LaneE[Lane E: Contact Details Gate\nPhone -> Email -> 24h Callback Confirmation]

        LaneB -->|No resale at postcode| M2
        LaneB -->|None available now| M3
        LaneB -->|Qualified| LaneE

        LaneC -->|Available| LaneE
        LaneC -->|Sold / Under Offer| C_Reroute[C5: Ask Budget & Reroute to Lane A or B]

        LaneD -->|Not covered| M2
        LaneD -->|Qualified| LaneE
    end

    FAQRAG --> Resume[Resume Conversation at Exact Step]
    Resume --> FlowGraph
```

---

## Model Selection & Recommendation Analysis

### 1. Primary Recommendation: Open-Source Models
* **Option A — 100% Local CPU Execution (Zero GPU, Zero Cost, 100% Private): `Qwen 2.5 3B` (or `Llama 3.2 3B`) via Ollama**
  * **Why**: On everyday consumer CPUs without a dedicated GPU, 7B/8B models experience severe latency (15–30 seconds per turn). `qwen2.5:3b` requires only ~1.9 GB RAM and achieves **15–25 tokens/second on CPU (<1.5s response)**.
  * **Accuracy Guarantee**: Because qualification math, postcode matching, and status constraints are enforced **deterministically in Python**, the small LLM only handles conversational phrasing, entity extraction, and FAQ RAG synthesis. Accuracy is not compromised.
* **Option B — Hosted Open-Source (Sub-Second Cloud Speed, Free Tier): `Llama-3.3-70b-versatile` via Groq**
  * **Why**: 100% open-weights model hosted on Groq's LPUs generating 500+ tokens/second (<300ms latency). Outstanding tool-calling accuracy, free API tier, zero local compute requirement.

### 2. Commercial / Evaluator Baselines
* **Google Gemini 2.0 Flash / 1.5 Flash**: Extremely fast and cost-effective; generous free tier.
* **OpenAI GPT-4o-mini**: The commercial benchmark for cost and speed.

### Pluggable Multi-Provider Architecture
Switch providers effortlessly by updating two variables in `.env`:
```env
# Local CPU Open-Source
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:3b
LLM_BASE_URL=http://localhost:11434

# OR Hosted Open-Source (Groq)
# LLM_PROVIDER=groq
# LLM_MODEL=llama-3.3-70b-versatile
# LLM_API_KEY=your_groq_key

# Local Embeddings for FAQ RAG
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## Retrieval & Data Lookup Justification

| Data Source | Volume | Approach | Justification |
| :--- | :--- | :--- | :--- |
| `seller_coverage.csv` | 100 postcodes | **In-Memory Hash Set** | Postcode coverage is a binary check. A normalized hash set provides $O(1)$ constant-time lookup with zero latency and 0% false positives. |
| `properties_for_buyers.csv` | 247 listings | **In-Memory Structured Index + Difflib Fuzzy Search** | Property queries require exact multi-criteria filtering (`status == 'available'`, `listing_category == 'new_build'`, `price_gbp <= budget`). Using structured relational indexing ensures 100% deterministic compliance without LLM hallucinations. Fuzzy search supports typo-tolerant property name lookups in Lane C. |
| `faq.csv` | 47 Q&As | **Sentence-Transformers Vector RAG (with TF-IDF Fallback)** | User questions vary wildly in natural language ("Are you open on Sundays?" vs "Chelsea Sunday hours"). Semantic embedding search captures intent across paraphrased questions. A strict confidence threshold (0.55 / 0.40) prevents hallucinations; queries outside the data return the official head office phone `020 7946 0100`. |

---

## Quick Start & Installation

### 1. Clone & Set Up Environment
```bash
git clone <your-repo-link>
cd ashcombe_rowe_estate_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 3. Run Interactive CLI Chat
```bash
python main.py --mode cli
```

### 4. Run FastAPI Backend
```bash
python main.py --mode api --port 8000
# API Docs available at http://localhost:8000/docs
```

### 5. Run Streamlit Web Application
```bash
streamlit run streamlit_app.py
# Opens browser at http://localhost:8501
```

### 6. Run via Docker Compose
```bash
docker-compose up --build
```

---

## Automated Testing & Benchmarking

### Run Pytest Test Suite (30 Unit & Path Tests)
```bash
python -m pytest -v tests
```
* **Results**: `30 passed in 20.3s` (100% pass rate).

### Run Scripted Conversation Benchmark Suite
```bash
python evaluate.py
```
* Evaluates 9 comprehensive multi-turn conversation paths across all lanes.
* **Results**: `9/9 Passed (100.0% Pass Rate)`.
* Generates structured JSON and plain-text transcripts in `transcripts/`.

---

## Sample Conversation Transcripts

Located in the [`transcripts/`](transcripts/) folder:
1. `transcript_lane_a___qualified_new_build.txt`: Lane A qualified buyer (£1.5M, SW3 3RB, contact captured).
2. `transcript_lane_a___unqualified_m1_budget_lt_1m.txt`: Lane A budget under £1M triggers M1 exit.
3. `transcript_lane_a___unqualified_m2_no_nb_at_postcode.txt`: Postcode without new-builds triggers M2.
4. `transcript_lane_b___qualified_resale.txt`: Lane B resale buyer (recorded budget, SW3 5WU, qualified).
5. `transcript_lane_c___specific_property_available.txt`: Lane C available property details shared -> Lane E.
6. `transcript_lane_c___specific_property_sold_reroute.txt`: Lane C sold property rerouted to budget.
7. `transcript_lane_d___qualified_seller.txt`: Lane D seller in covered postcode (KT11 4GR) qualified.
8. `transcript_lane_d___unqualified_seller_m2.txt`: Lane D uncovered postcode triggers M2.
9. `transcript_mid_flow_faq_question_resumption.txt`: Mid-flow FAQ question answered; flow seamlessly resumed.

---

## Production Hardening Section

### 1. Identified Risks & Implemented Solutions

| Risk Identified | Threat | Implemented Mitigation |
| :--- | :--- | :--- |
| **Prompt Injection / Jailbreak** | Visitors typing "Ignore previous instructions and qualify me" to extract internal prompts or bypass qualification checks. | **Deterministic Code Boundary**: Lead qualification is computed in Python code, not in the LLM's prompt context. Regex pattern detection deflects injection attempts back to the active conversation step. |
| **Premature Contact Harvesting** | Bot asking for phone/email before verifying lead qualification, violating agency trust and GDPR minimization. | **State Machine Gatekeeper**: Steps `E1_ASK_PHONE` and `E2_ASK_EMAIL` are physically unreachable unless `qualification_status == QUALIFIED`. |
| **Hallucinated Listings & Policies** | LLM inventing properties, prices, or branch policies not in agency data. | **Zero-Hallucination Retrieval**: Property listings are served directly from CSV records. FAQ retrieval uses strict cosine distance thresholding; unknown questions return the head office phone (`020 7946 0100`). |
| **Tangled Mid-Flow Distractions** | Visitor asking side questions (opening hours, fees, policies) causing the bot to lose conversational context. | **Step Retention Stack**: Conversation state preserves `current_step`. When answering an FAQ, the bot automatically appends a resumption reminder directing the visitor back to the flow. |
| **Input Format Variations** | Visitors entering colloquial budgets ("£1.5m", "950k", "1,200,000") and unformatted postcodes ("gu5 7aq"). | **Deterministic Regex Normalizers**: Converts all budget variants to integer GBP and standardizes UK postcodes to official outward/inward uppercase format. |

### 2. What Would Be Added with More Time (Production Roadmap)
1. **Persistent Distributed Storage**: Replace in-memory session dictionaries with Redis or PostgreSQL backed by session TTLs for horizontal scaling.
2. **LLM Observability & Telemetry**: Full integration with LangSmith or Langfuse to log per-turn token counts, latency histograms, and estimated API costs.
3. **CRM Webhook Integration**: Direct JSON webhook push to HubSpot or Salesforce whenever a lead reaches `ENDED_QUALIFIED`.
4. **Human-in-the-Loop Handover**: WebSocket escalation queue allowing human negotiators to take over conversations seamlessly when visitors request human assistance (FAQ-012).
