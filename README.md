# Spatial Optimization Conversational Agent

An academic research tool for solving facility location problems using conversational AI powered by Claude.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This application combines cutting-edge optimization algorithms with natural language AI to help urban planners, researchers, and administrators solve complex spatial optimization problems through simple conversation.

### Key Features

- 🗣️ **Conversational Interface**: Describe your problem in natural language - no coding required
- 🗺️ **Interactive Visualization**: See your solutions on beautiful interactive maps
- 🎯 **Multiple Problem Types**: P-Median, P-Center, MCLP, LSCP, and more
- 📊 **Rich Analytics**: Comprehensive metrics and performance indicators
- 🔬 **Academic Rigor**: Full citations, mathematical formulations, and methodology documentation
- 🔌 **Extensible Architecture**: Easy to add new problem types
- 💾 **Export Capabilities**: GeoJSON, CSV, PDF reports, and more

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Google Gemini API key (get one at [Google AI Studio](https://aistudio.google.com/))
- Gurobi license (optional - will fall back to PuLP if not available)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd spoptv2
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Unix/MacOS
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
echo GEMINI_API_KEY=your_api_key_here > .env
```

Alternatively, create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your_api_key_here"
```

5. **Generate test data** (optional)
```bash
python tests/generate_test_data.py
```

6. **Run the application**
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage

### 1. Upload Data

Upload your geospatial data files in the sidebar:
- **Demand points**: Locations with demand/population (GeoJSON, Shapefile, CSV with coordinates)
- **Candidate sites**: Potential facility locations (same formats)

### 2. Describe Your Problem

Simply describe what you want to achieve:

**Example conversations:**
- "I need to locate 5 fire stations to minimize response times"
- "Where should I place 3 warehouses to minimize average shipping distance?"
- "I want to maximize coverage within 5km using 4 facilities"
- "What's the minimum number of ambulance stations needed to cover all areas within 10 minutes?"

### 3. Review Results

- View the solution on an interactive map
- Analyze performance metrics
- Export results in multiple formats
- Iterate and refine your solution

## 🧮 Supported Problem Types

### P-Median Problem
**Objective**: Minimize total or average weighted distance

**Use cases**: Warehouse location, distribution centers, service facilities

**Example**: "Locate 5 libraries to minimize average distance for residents"

### P-Center Problem
**Objective**: Minimize maximum distance (minimax/equity)

**Use cases**: Emergency services, disaster relief, equal access requirements

**Example**: "Place 3 fire stations to minimize worst-case response time"

### Maximum Covering Location Problem (MCLP)
**Objective**: Maximize demand covered within a service radius

**Use cases**: Retail placement, emergency coverage, facility budgeting

**Example**: "Maximize population served within 5km using 4 health clinics"

### Location Set Covering Problem (LSCP)
**Objective**: Minimize facilities needed for full coverage

**Use cases**: Minimum infrastructure deployment, cost optimization

**Example**: "What's the minimum number of cell towers to cover the entire city?"

## 📁 Project Structure

```
spoptv2/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .env.example                    # Example environment variables
├── config/
│   └── settings.py                 # Configuration settings
├── solvers/
│   ├── base_solver.py              # Abstract base class
│   ├── registry.py                 # Problem registry
│   ├── p_median_solver.py          # P-Median implementation
│   ├── p_center_solver.py          # P-Center implementation
│   ├── mclp_solver.py              # MCLP implementation
│   └── lscp_solver.py              # LSCP implementation
├── agent/
│   ├── conversation_manager.py     # Claude API integration
│   └── prompts.py                  # System prompts
├── utils/
│   ├── data_processor.py           # Data loading and validation
│   ├── distance_calculator.py      # Distance matrix computation
│   ├── visualizer.py               # Map generation
│   └── export_handler.py           # Export functionality
├── tests/
│   ├── test_solvers.py             # Unit tests
│   ├── generate_test_data.py       # Test data generation
│   └── test_data/                  # Sample datasets
└── docs/
    └── architecture.md             # System architecture
```

## 🔬 Academic Use

This tool is designed for academic research and includes:

- **Mathematical formulations** for each problem type
- **Academic references** in APA format
- **Computational complexity** analysis
- **Algorithm documentation** with citations
- **Reproducible solutions** (all parameters saved)

### Citations

When using this tool in research, please cite the relevant papers listed in each solver's metadata. Access citations through:
```python
from solvers.registry import problem_registry
solver = problem_registry.get_problem("p-median")
references = solver.get_metadata()["academic_refs"]
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=solvers --cov=utils --cov=agent
```

## 🔧 Configuration

### Solver Settings

Edit `config/settings.py` to adjust:
- Preferred solver (Gurobi vs PuLP)
- Time limits
- MIP gap tolerances
- Distance calculation methods
- Visualization defaults

### Distance Metrics

Supported distance metrics:
- **Euclidean**: Straight-line distance (default)
- **Manhattan**: Grid-based distance
- **Network**: Road network distance (requires OSMnx - future feature)

## 📊 Export Options

Export your solutions in multiple formats:

- **GeoJSON**: Selected facilities with metadata
- **CSV**: Assignments and metrics
- **Shapefile**: For use in GIS software
- **PDF Report**: Comprehensive solution documentation

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Adding New Problem Types

1. Create a new solver class inheriting from `SpatialOptimizationProblem`
2. Implement all required abstract methods
3. Add to registry in `solvers/registry.py`
4. Write tests in `tests/`
5. Document in `docs/problems/`

See `docs/architecture.md` for detailed instructions.

## 🐛 Troubleshooting

### Common Issues

**"ANTHROPIC_API_KEY not found"**
- Ensure you've set the API key in `.env` or `.streamlit/secrets.toml`

**"Gurobi not available"**
- This is fine! The system will automatically use PuLP as a fallback
- For better performance on large problems, install Gurobi with a license

**"No module named 'geopandas'"**
- Run `pip install -r requirements.txt` to install all dependencies

**Map not displaying**
- Check that your data has valid geometries
- Ensure CRS is set correctly (should auto-detect to EPSG:4326)

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Google Gemini for conversational AI
- Gurobi/PuLP for optimization engines
- spopt and PySAL for spatial optimization algorithms
- Streamlit for the web framework
- OpenStreetMap for base maps

## 📧 Contact

For questions, issues, or contributions:
- Open an issue on GitHub
- Email: [your-email@example.com]

## 🗺️ Roadmap

- [ ] Network distance calculations using OSMnx
- [ ] Capacitated facility location problems
- [ ] Multi-objective optimization
- [ ] Real-time data integration (Census API, etc.)
- [ ] Stochastic optimization models
- [ ] Time-dependent problems
- [ ] 3D visualization options

---

**Built with ❤️ for urban planning and spatial optimization research**

