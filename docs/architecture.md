# System Architecture

## Overview

The Spatial Optimization Conversational Agent follows a modular, plugin-based architecture that separates concerns and enables easy extensibility.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                     │
│  (app.py - User Interface, File Upload, Visualization)  │
└─────────────────┬───────────────────────┬───────────────┘
                  │                       │
                  ▼                       ▼
         ┌────────────────┐      ┌──────────────────┐
         │ Conversation   │      │  Data Processor  │
         │   Manager      │      │   & Visualizer   │
         └────────┬───────┘      └──────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Gemini API    │
         │   (Google)     │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │    Problem     │
         │   Registry     │
         └────────┬───────┘
                  │
         ┌────────┴──────────────────────────┐
         ▼                ▼                   ▼
    ┌────────┐      ┌────────┐         ┌────────┐
    │P-Median│      │P-Center│   ...   │  LSCP  │
    │ Solver │      │ Solver │         │ Solver │
    └────┬───┘      └────┬───┘         └────┬───┘
         │               │                   │
         └───────────────┴───────────────────┘
                         ▼
                ┌─────────────────┐
                │  Optimization   │
                │ Engines (Gurobi │
                │   /PuLP)        │
                └─────────────────┘
```

## Component Details

### 1. Streamlit Frontend (`app.py`)

**Responsibilities:**
- User interface and interaction
- File upload handling
- Chat interface
- Map visualization
- Metrics display
- Export functionality

**Key Features:**
- Session state management for conversation history
- Real-time map updates
- Interactive chat with Google Gemini
- Responsive layout (two-column design)

**State Management:**
```python
session_state = {
    "messages": [],              # Chat history
    "problem_state": {          # Current optimization problem
        "problem_type": str,
        "parameters": dict,
        "constraints": dict,
        "data": dict,
        "solution": dict
    },
    "conversation_manager": ConversationManager,
    "data_processor": DataProcessor,
    "map_visualizer": MapVisualizer
}
```

### 2. Conversation Manager (`agent/conversation_manager.py`)

**Responsibilities:**
- Interface with Gemini API
- Maintain conversation context
- Parse user intent
- Extract structured actions (JSON)
- Update problem state

**Design Pattern:** Facade pattern for AI interactions

**Key Methods:**
- `chat()`: Send message with full context
- `_prepare_messages()`: Build message array with state
- `_parse_response()`: Extract actions from Gemini's response

**Context Management:**
- Includes full conversation history in every API call (Gemini is stateless)
- Embeds current problem state in messages
- Provides data summaries for awareness

### 3. Problem Registry (`solvers/registry.py`)

**Responsibilities:**
- Register and manage all problem types
- Problem discovery and retrieval
- Natural language problem type inference
- Metadata aggregation

**Design Pattern:** Registry + Singleton

**Key Features:**
- Auto-discovery of problem classes
- Keyword-based problem detection
- Category-based organization
- Academic reference aggregation

### 4. Base Solver Class (`solvers/base_solver.py`)

**Responsibilities:**
- Define interface for all optimization problems
- Ensure consistency across problem types
- Provide default implementations where appropriate

**Design Pattern:** Template Method + Strategy

**Required Methods:**
```python
class SpatialOptimizationProblem(ABC):
    @abstractmethod
    def get_metadata() -> Dict
    
    @abstractmethod
    def get_conversation_prompts() -> Dict
    
    @abstractmethod
    def get_required_data() -> Dict
    
    @abstractmethod
    def validate_parameters() -> Tuple[bool, str]
    
    @abstractmethod
    def solve() -> Dict
    
    @abstractmethod
    def explain_solution() -> str
    
    def get_visualization_config() -> Dict  # Optional override
    def sensitivity_analysis() -> List      # Optional override
```

### 5. Individual Solvers

Each solver implements `SpatialOptimizationProblem`:

**P-Median Solver** (`solvers/p_median_solver.py`)
- Minimize total/average weighted distance
- MIP formulation with Gurobi/PuLP
- Supports custom constraints

**P-Center Solver** (`solvers/p_center_solver.py`)
- Minimize maximum distance (minimax)
- MIP with minimax objective variable
- Equity-focused applications

**MCLP Solver** (`solvers/mclp_solver.py`)
- Maximize coverage within threshold
- Coverage matrix pre-computation
- Fixed number of facilities

**LSCP Solver** (`solvers/lscp_solver.py`)
- Minimize facilities for full coverage
- Set covering formulation
- Feasibility checking

### 6. Utility Modules

**Data Processor** (`utils/data_processor.py`)
- Load various geospatial formats (GeoJSON, Shapefile, CSV)
- Validate geometries and required fields
- CRS standardization
- Data type identification

**Distance Calculator** (`utils/distance_calculator.py`)
- Compute distance matrices
- Multiple metrics (Euclidean, Manhattan, Network)
- Coverage matrix generation
- CRS handling and projection

**Map Visualizer** (`utils/visualizer.py`)
- Create interactive Folium maps
- Layered visualization (demand, candidates, facilities)
- Assignment lines and service areas
- Legend and layer control

**Export Handler** (`utils/export_handler.py`)
- Export to GeoJSON, CSV, Shapefile
- Generate PDF reports
- Session saving for reproducibility

## Data Flow

### 1. Initial Setup
```
User uploads data → DataProcessor → Validation → Session State
```

### 2. Problem Identification
```
User describes problem → ConversationManager → Gemini API
    → Problem Registry → Infer problem type → Update state
```

### 3. Parameter Collection
```
Gemini asks questions → User responds → Parameters extracted
    → Validate parameters → Update state
```

### 4. Optimization Trigger
```
Gemini returns JSON action → Extract problem_type & parameters
    → Get solver from registry → Prepare data → Solve
    → Update state with solution
```

### 5. Solution Visualization
```
Solution + Data → MapVisualizer → Folium map → Streamlit display
Solution → Metrics dashboard → Streamlit metrics
```

## Design Patterns Used

### 1. **Template Method**
- `SpatialOptimizationProblem` defines algorithm structure
- Concrete solvers implement specific steps
- Ensures consistency while allowing customization

### 2. **Strategy Pattern**
- Different solver strategies for different problem types
- Interchangeable at runtime
- Selected based on problem identification

### 3. **Registry Pattern**
- Central problem registry
- Auto-discovery of problem types
- Decoupled registration and usage

### 4. **Facade Pattern**
- `ConversationManager` simplifies Gemini API interaction
- Hides complexity of context management
- Provides simple `chat()` interface

### 5. **Singleton Pattern**
- Single global `problem_registry` instance
- Ensures consistent problem catalog

## Extension Points

### Adding New Problem Types

1. **Create Solver Class**
```python
# solvers/my_problem_solver.py
from .base_solver import SpatialOptimizationProblem

class MyProblemSolver(SpatialOptimizationProblem):
    def get_metadata(self):
        return {
            "name": "My Problem",
            "short_name": "my-problem",
            # ... complete metadata
        }
    
    # Implement all required methods
```

2. **Register in Registry**
```python
# solvers/registry.py
def _register_default_problems(self):
    from .my_problem_solver import MyProblemSolver
    self.register(MyProblemSolver)
```

3. **That's it!** The problem is now:
- Available in the conversation
- Detectable by keywords
- Solvable through the interface

### Adding New Distance Metrics

1. Add method to `DistanceCalculator`
2. Update `calculate_distance_matrix()` switch
3. Document assumptions

### Adding New Export Formats

1. Add method to `ExportHandler`
2. Add button in `app.py`
3. Handle file download

## Performance Considerations

### Optimization Performance
- **Small problems** (<100 demand, <50 candidates): < 5 seconds
- **Medium problems** (<500 demand, <200 candidates): < 30 seconds
- **Large problems**: May require time limit adjustments

### Optimization Strategies
- Gurobi preferred for speed (commercial license required)
- PuLP fallback (free, open-source)
- Time limits prevent hanging
- MIP gap allows near-optimal solutions

### Map Rendering
- Folium handles up to ~1000 features well
- Larger datasets may need clustering
- Layer control for selective display

- Conversation Performance
- Gemini API calls: ~1-3 seconds
- Full context included (no memory between calls)
- Stateless design ensures consistency

## Security Considerations

1. **API Key Management**
   - Never commit API keys
   - Use `.env` or Streamlit secrets
   - Validate on startup

2. **File Upload Validation**
   - Check file types
   - Validate geometries
   - Size limits enforced

3. **Input Validation**
   - Parameter validation for each problem
   - Constraint validation
   - Error handling for malformed data

## Testing Strategy

### Unit Tests
- Individual solver correctness
- Distance calculations
- Data processing
- Problem registry

### Integration Tests
- End-to-end conversation flows
- Data → Solve → Visualize pipeline
- Export functionality

### Test Data
- Synthetic datasets (controlled, reproducible)
- Real-world examples (anonymized)
- Edge cases (empty, invalid, infeasible)

## Deployment Considerations

### Local Deployment
```bash
streamlit run app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Set secrets in dashboard
4. Deploy

### Docker Deployment
```dockerfile
FROM python:3.10
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

## Future Enhancements

1. **Network Distances**
   - Integrate OSMnx for road networks
   - Real-time routing APIs

2. **Advanced Visualization**
   - 3D elevation maps
   - Time-series animations
   - Comparative analysis views

3. **Scalability**
   - Database backend for large datasets
   - Async optimization for responsiveness
   - Batch processing mode

4. **Collaboration**
   - Multi-user sessions
   - Shared workspaces
   - Version control for solutions

## References

- Daskin, M. S. (2013). Network and discrete location: models, algorithms, and applications. John Wiley & Sons.
- Streamlit Documentation: https://docs.streamlit.io
- Google Gemini API: https://ai.google.dev
- Gurobi Optimizer: https://www.gurobi.com/documentation
- PuLP Documentation: https://coin-or.github.io/pulp

