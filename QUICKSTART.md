# Quick Start Guide

Get up and running with the Spatial Optimization Conversational Agent in 5 minutes!

## Prerequisites

- Python 3.10 or higher
- Google Gemini API key ([Get one here](https://aistudio.google.com/))

## Installation

### Option 1: Automated Setup (Recommended)

**Windows:**
```bash
run.bat
```

**Linux/MacOS:**
```bash
chmod +x run.sh
./run.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Set up configuration files
- Start the application

### Option 2: Manual Setup

1. **Create virtual environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/MacOS
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run setup:**
```bash
python setup.py
```

4. **Configure API key:**

Edit `.env` file:
```bash
GEMINI_API_KEY=your-gemini-key-here
```

Or edit `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-gemini-key-here"
```

5. **Start the application:**
```bash
streamlit run app.py
```

## First Use

### 1. Upload Data

The application needs two types of geospatial data:

**Option A: Use Test Data**
```bash
python tests/generate_test_data.py
```

Then upload from `tests/test_data/`:
- `sample_demand.geojson` (demand points)
- `sample_candidates.geojson` (candidate sites)

**Option B: Use Your Own Data**

Upload files in these formats:
- GeoJSON (`.geojson`, `.json`)
- Shapefile (`.zip` containing .shp, .shx, .dbf, .prj)
- CSV with coordinates (needs `lat`/`lon` or `x`/`y` columns)

### 2. Describe Your Problem

Start a conversation! Try these examples:

**P-Median (Minimize Average Distance):**
> "I need to locate 5 warehouses to minimize average shipping distance"

**P-Center (Minimize Worst Case):**
> "Place 3 fire stations to minimize maximum response time"

**MCLP (Maximize Coverage):**
> "I want to maximize population covered within 5km using 4 facilities"

**LSCP (Minimize Facilities):**
> "What's the minimum number of ambulance stations to cover everyone within 10 minutes?"

### 3. Follow the Conversation

The AI will:
- Confirm the problem type
- Ask for parameters
- Verify your data
- Run the optimization
- Explain the results

### 4. View Results

- **Map**: See selected facilities and demand assignments
- **Metrics**: Analyze performance indicators
- **Export**: Download results in GeoJSON, CSV, or PDF

## Example Session

```
You: I need to find the best locations for 3 new libraries

AI: Great! It sounds like you want to minimize the average distance 
residents need to travel to a library. This is a P-Median problem.

Before we proceed:
1. Have you uploaded data showing where residents live?
2. Do you have potential library sites?

You: [uploads population.geojson and sites.geojson]

AI: Perfect! I see:
- 1,250 population points
- 25 candidate library sites

Should I proceed with locating 3 libraries to minimize 
average distance?

You: Yes

AI: ✅ Optimization Complete!

P-Median Solution Summary:
- Located 3 libraries optimally
- Average distance: 1,247 meters
- Total weighted distance: 45,892

The selected facilities minimize overall access distance...
```

## Troubleshooting

### "GEMINI_API_KEY not found"
→ Make sure you've set the API key in `.env` or `.streamlit/secrets.toml`

### "No module named 'geopandas'"
→ Run: `pip install -r requirements.txt`

### "Gurobi not available"
→ This is fine! The system automatically uses PuLP (free alternative)
→ For better performance on large problems, install Gurobi with a license

### Map not displaying
→ Check that your data has valid geometries
→ Ensure files are in proper geospatial formats

### Slow optimization
→ Try with smaller datasets first
→ Consider using Gurobi for large problems
→ Adjust time limits in `config/settings.py`

## Tips

1. **Start Small**: Use test data or small datasets first
2. **Be Specific**: Clearly describe your optimization goal
3. **Iterate**: Try different parameters and compare solutions
4. **Export Early**: Save solutions you like
5. **Ask Questions**: The AI can explain concepts and trade-offs

## Next Steps

- **Learn More**: Read `README.md` for comprehensive documentation
- **Understand Architecture**: Check `docs/architecture.md`
- **Add Problems**: See `CONTRIBUTING.md` for extending the system
- **Get Help**: Open an issue on GitHub

## Sample Problems to Try

### Urban Planning
- "Locate 5 recycling centers to minimize collection distance"
- "Where should we build 3 community centers for equal access?"

### Emergency Services
- "Place 4 fire stations to minimize worst-case response time"
- "What's the minimum number of ambulances to cover the city in 8 minutes?"

### Business
- "Locate 6 retail stores to maximize customer reach within 10km"
- "Find optimal distribution center locations for 3 facilities"

### Public Health
- "Place 5 vaccination sites to maximize population covered within 5km"
- "Minimize hospitals needed to ensure everyone is within 15 minutes"

## Resources

- **Documentation**: `docs/architecture.md`
- **Examples**: `tests/test_data/`
- **API Reference**: `solvers/base_solver.py`
- **Community**: GitHub Issues

## Support

Having issues? 

1. Check this guide
2. Read the README.md
3. Review common errors above
4. Open an issue on GitHub
5. Email: [your-email@example.com]

---

**Happy Optimizing! 🗺️**

