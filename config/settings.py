import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application configuration"""
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Model settings
    GEMINI_MODEL = "gemini-2.0-flash-exp"  # Latest Gemini Flash model
    MAX_TOKENS = 4096
    
    # Solver preferences
    PREFERRED_SOLVER = "gurobi"  # Falls back to pulp if Gurobi unavailable
    SOLVER_TIME_LIMIT = 300  # seconds
    MIP_GAP = 0.01  # 1% optimality gap
    
    # Performance settings
    ENABLE_DISTANCE_CACHING = True
    MAX_CACHE_SIZE = 10
    ENABLE_PERFORMANCE_LOGGING = True
    
    # Solver optimization settings
    GUROBI_PRESOLVE = 2  # Aggressive presolve
    GUROBI_CUTS = 2      # Aggressive cut generation
    GUROBI_HEURISTICS = 0.05  # 5% time on heuristics
    
    # Distance calculation
    DEFAULT_DISTANCE_METRIC = "euclidean"
    CRS_STANDARD = "EPSG:4326"  # WGS84 for lat/lon
    CRS_PROJECTED = "EPSG:3857"  # Web Mercator for distance calculations
    
    # File upload limits
    MAX_UPLOAD_SIZE_MB = 50
    ALLOWED_EXTENSIONS = [".geojson", ".json", ".csv", ".shp", ".zip"]
    
    # Visualization
    DEFAULT_MAP_ZOOM = 12
    FACILITY_MARKER_COLOR = "red"
    DEMAND_MARKER_COLOR = "blue"
    CANDIDATE_MARKER_COLOR = "gray"
    
    # Academic
    CITATION_STYLE = "APA"
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    TEMP_DIR = BASE_DIR / "temp"
    DOCS_DIR = BASE_DIR / "docs"
    TEST_DATA_DIR = BASE_DIR / "tests" / "test_data"
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set. Please set it in .env file or environment variables.")
        
        # Create directories if they don't exist
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.TEMP_DIR.mkdir(exist_ok=True)
        cls.TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        return True
    
    @classmethod
    def check_gurobi(cls) -> bool:
        """Check if Gurobi is available"""
        try:
            import gurobipy
            return True
        except ImportError:
            return False

settings = Settings()

