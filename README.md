# 🏥 Curalink — AI Medical Research Assistant

> A full-stack AI-powered Medical Research Assistant that retrieves, ranks, and synthesizes peer-reviewed publications and clinical trials into structured, personalized, research-backed insights.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          User Query                              │
│               (Disease + Intent + Patient Context)               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │ React Frontend │  (Vite + Tailwind CSS)
                    └───────┬────────┘
                            │ REST API
                    ┌───────▼────────┐
                    │ Express Backend │  (Node.js + MongoDB)
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  FastAPI AI    │  Query Expansion (LLM)
                    │   Service      │
                    └──┬──────┬──┬───┘
                       │      │  │
               ┌───────┘  ┌───┘  └──────────┐
        PubMed API  OpenAlex API  ClinicalTrials.gov v2
               │          │               │
               └──────────┴───────────────┘
                        50-300 candidates
                              │
                    ┌─────────▼──────────┐
                    │  Embedding + Rank   │  (all-MiniLM-L6-v2)
                    │  Top 6-8 selected   │  Semantic + Recency
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Llama-3-8B (Groq)  │  Structured Synthesis
                    └─────────┬──────────┘
                              │
                    Structured JSON Response:
                    - conditionOverview
                    - answer (markdown)
                    - publications[]
                    - clinicalTrials[]
```

## 🚀 Quick Start (Local Development)

### Prerequisites
- Node.js 18+
- Python 3.12+
- MongoDB running locally (or MongoDB Atlas URI)
- [Groq API Key](https://console.groq.com) — **free, 30s/request, open-source Llama-3**

---

### Step 1 — Configure API Keys

**AI Service** (`ai-service/.env`):
```env
GROQ_API_KEY=gsk_your_key_here
PORT=8000
```

**Backend** (`backend/.env`):
```env
PORT=5000
MONGODB_URI=mongodb://localhost:27017/curalink
AI_SERVICE_URL=http://localhost:8000
FRONTEND_URL=http://localhost:5173
NODE_ENV=development
```

---

### Step 2 — Start AI Service (Terminal 1)
```powershell
cd ai-service
venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3 — Start Backend (Terminal 2)
```powershell
cd backend
node server.js
```

### Step 4 — Start Frontend (Terminal 3)
```powershell
cd frontend
npm run dev
```

Open **http://localhost:5173** 🎉

---

## 🔑 Getting a Groq API Key (FREE)

1. Go to [https://console.groq.com](https://console.groq.com)
2. Create a free account
3. Go to API Keys → Create API Key
4. Paste it in `ai-service/.env` as `GROQ_API_KEY`

The free tier gives you **30 requests/minute** with **Llama-3-8B-8192**, which is more than enough for this demo.

---

## 🐳 Docker (All-in-One)

```bash
# Set your GROQ_API_KEY in ai-service/.env first
docker-compose up --build
```

---

## 🧠 AI Pipeline Details

| Step | What happens |
|------|-------------|
| Query Expansion | Llama-3 creates optimized medical search terms |
| Broad Retrieval | 100 PubMed + 100 OpenAlex + 50 ClinicalTrials = **~250 candidates** |
| Deduplication | Remove duplicate titles; dedupe trials by NCT ID |
| Embedding | all-MiniLM-L6-v2 encodes query + all abstracts (384-d vectors) |
| Ranking | 60% cosine similarity + 30% recency + 10% source credibility |
| Top Selection | Best 8 publications + best 6 clinical trials |
| LLM Synthesis | Llama-3-8B generates `conditionOverview` + `answer` with citations |
| Context Memory | Last 6 conversation turns injected into LLM context |

---

## 📋 Demo Queries

- "Latest treatment for lung cancer"
- "Clinical trials for diabetes"
- "Top researchers in Alzheimer's disease"
- "Recent studies on heart disease"
- "Deep Brain Stimulation for Parkinson's disease" (with patient context!)

---

## 📁 Project Structure

```
Curalink-AI-Medical-Research-Assistant/
├── frontend/        # React + Vite + Tailwind CSS
│   └── src/
│       ├── components/   # ChatMessage, PublicationCard, ClinicalTrialCard, Sidebar
│       ├── pages/        # ChatPage
│       ├── store/        # Zustand chat store
│       └── api/          # Axios API client
├── backend/         # Node.js + Express + MongoDB
│   ├── models/      # Conversation schema
│   └── routes/      # /chat, /history, /health
├── ai-service/      # Python FastAPI
│   ├── core/        # embeddings.py, ranker.py, llm_service.py
│   ├── retrievers/  # pubmed.py, openalex.py, clinical_trials.py
│   └── routers/     # research.py (main pipeline)
└── docker-compose.yml
```
