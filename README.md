# **SOCA: A Conversational Agent for Facility Location Problems**

Application for conversational facility-location optimization using geospatial data and mixed-integer programming.

## Scope

Supported models:

- P-Median
- P-Center
- MCLP (including budget/capacity/probabilistic/multi-coverage variants)
- LSCP

Core capabilities:

- Natural-language problem setup via Gemini
- Geospatial input handling (CSV/GeoJSON/Shapefile)
- Interactive map output and result export

## Repository Contents

- `app.py`: Streamlit entry point
- `agent/`: conversation and prompt logic
- `solvers/`: optimization model implementations
- `utils/`: data processing, distance, visualization, export helpers
- `datasets/`: included datasets used for experiments/examples
- `tests/`: unit and integration-oriented tests

## Requirements

- Python 3.10+
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

## License

MIT License. See `LICENSE`.

## Contact

For issues and questions, use repository issues or contact: `mahad.imran29@gmail.com`