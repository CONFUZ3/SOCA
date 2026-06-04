# **SOCA: A Conversational Agent for Facility Location Problems**

Application for conversational facility-location optimization using geospatial data and mixed-integer programming.

## Scope

Supported models:

- P-Median
- P-Center
- MCLP 
- LSCP

Core capabilities:

- Natural-language problem setup via LLM
- Geospatial input handling (CSV/GeoJSON/Shapefile)
- Interactive map output and result export

## Architecture

The product is a **FastAPI backend + Next.js (React) frontend**. The legacy Streamlit `app.py` still works for local use.

## Repository Contents

- `frontend/`: Next.js 16 + React UI (MapLibre map, chat, sidebar)
- `backend/`: FastAPI server (REST/SSE API under `/api/*`)
- `agent/`: Google ADK conversational agent and tools
- `solvers/`: optimization model implementations
- `utils/`: data fetching, distance, visualization, export helpers
- `app.py`: legacy Streamlit entry point
- `datasets/`: included datasets used for experiments/examples
- `tests/`: unit and integration-oriented tests

## Requirements

- Python 3.10+
- Node.js 20+ (for the React frontend)
- `pip` and virtual environment tooling
- Gemini API key
- Optional: Gurobi license (PuLP fallback is supported)

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

Create a local env file (do not commit secrets):

```bash
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env` or in `.streamlit/secrets.toml`.

## Run

Start the FastAPI backend:

```bash
uvicorn backend.main:app --reload --port 8000
```

Start the Next.js frontend (in a second terminal):

```bash
cd frontend
npm install
npm run dev
```

App: `http://localhost:3000` · API: `http://localhost:8000`

### Legacy Streamlit app

```bash
streamlit run app.py
```

Default local URL: `http://localhost:8501`

## Data

The `datasets/` directory is intentionally included for reproducibility.

Expected inputs:

- Demand points with optional demand weights
- Candidate facility locations

## Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=solvers --cov=utils --cov=agent
```


## Contact
For issues and questions, use repository issues or contact: `mahad.imran29@gmail.com`
