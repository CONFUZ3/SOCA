from typing import Dict, List, Type, Optional, Any
from .base_solver import SpatialOptimizationProblem
import logging

logger = logging.getLogger(__name__)

class ProblemRegistry:
    """
    Central registry for all spatial optimization problems.
    Supports dynamic problem discovery and registration.
    """
    
    def __init__(self):
        self._problems: Dict[str, SpatialOptimizationProblem] = {}
        self._metadata_cache: Dict[str, Dict] = {}
        self._keyword_index: Dict[str, List[str]] = {}  # keyword -> list of problem short_names
        self._register_default_problems()
    
    def register(self, problem_class: Type[SpatialOptimizationProblem]):
        """Register a new problem type"""
        try:
            # Instantiate the problem
            problem_instance = problem_class()
            
            # Get and cache metadata
            metadata = problem_instance.get_metadata()
            short_name = metadata.get('short_name')
            
            if not short_name:
                raise ValueError(f"Problem class {problem_class.__name__} must provide 'short_name' in metadata")
            
            # Store problem instance
            self._problems[short_name] = problem_instance
            self._metadata_cache[short_name] = metadata
            
            # Index keywords for detection
            keywords = metadata.get('keywords', [])
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower not in self._keyword_index:
                    self._keyword_index[keyword_lower] = []
                self._keyword_index[keyword_lower].append(short_name)
            
            logger.info(f"Registered problem: {metadata.get('name')} ({short_name})")
            
        except Exception as e:
            logger.error(f"Failed to register problem {problem_class.__name__}: {e}")
            raise
    
    def get_problem(self, short_name: str) -> Optional[SpatialOptimizationProblem]:
        """Get problem instance by short name"""
        return self._problems.get(short_name)
    
    def list_problems(self) -> List[Dict[str, Any]]:
        """List all registered problems with full metadata"""
        return list(self._metadata_cache.values())
    
    def infer_problem_type(self, user_message: str) -> Optional[str]:
        """
        Infer problem type from natural language.
        Uses keyword matching and phrase detection.
        Returns short_name of most likely problem or None.
        """
        user_message_lower = user_message.lower()
        
        # Score each problem based on keyword matches
        scores = {}
        for short_name in self._problems.keys():
            score = 0
            metadata = self._metadata_cache[short_name]
            
            # Check keywords
            for keyword in metadata.get('keywords', []):
                if keyword.lower() in user_message_lower:
                    score += 2
            
            # Check problem name
            if metadata.get('name', '').lower() in user_message_lower:
                score += 5
            
            if short_name in user_message_lower:
                score += 5
            
            if score > 0:
                scores[short_name] = score
        
        # Return highest scoring problem
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def get_problems_by_category(self, category: str) -> List[str]:
        """Get all problems in a category (coverage, distance, etc.)"""
        return [
            short_name 
            for short_name, metadata in self._metadata_cache.items()
            if metadata.get('category', '').lower() == category.lower()
        ]
    
    def get_academic_references(self) -> Dict[str, List[str]]:
        """Aggregate all academic references for citations"""
        references = {}
        for short_name, metadata in self._metadata_cache.items():
            references[short_name] = metadata.get('academic_refs', [])
        return references
    
    def search_problems(self, query: str) -> List[Dict[str, Any]]:
        """Search problems by keyword, use case, or description"""
        query_lower = query.lower()
        results = []
        
        for short_name, metadata in self._metadata_cache.items():
            # Search in multiple fields
            searchable_text = " ".join([
                metadata.get('name', ''),
                metadata.get('description', ''),
                " ".join(metadata.get('keywords', [])),
                " ".join(metadata.get('typical_use_cases', []))
            ]).lower()
            
            if query_lower in searchable_text:
                results.append(metadata)
        
        return results
    
    def _register_default_problems(self):
        """Register all default problem types"""
        # Import and register problem solvers
        # This will be populated as we create each solver
        try:
            from .p_median_solver import PMedianSolver
            self.register(PMedianSolver)
        except ImportError:
            pass
        
        try:
            from .p_center_solver import PCenterSolver
            self.register(PCenterSolver)
        except ImportError:
            pass
        
        try:
            from .mclp_solver import MCLPSolver
            self.register(MCLPSolver)
        except ImportError:
            pass
        
        try:
            from .lscp_solver import LSCPSolver
            self.register(LSCPSolver)
        except ImportError:
            pass

# Global singleton
problem_registry = ProblemRegistry()

