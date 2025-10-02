# Contributing to Spatial Optimization Conversational Agent

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Adding New Problem Types](#adding-new-problem-types)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a positive learning environment
- Academic integrity and proper attribution

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/spoptv2.git
   cd spoptv2
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/original-owner/spoptv2.git
   ```

## Development Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install development dependencies:
   ```bash
   pip install pytest pytest-cov black flake8 mypy
   ```

4. Set up pre-commit hooks (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## How to Contribute

### Reporting Bugs

- Check if the bug has already been reported
- Use the issue template
- Include:
  - Clear description
  - Steps to reproduce
  - Expected vs actual behavior
  - Environment details (OS, Python version, etc.)
  - Sample data if relevant

### Suggesting Enhancements

- Check if the enhancement has been suggested
- Provide:
  - Clear use case
  - Why it's valuable
  - Potential implementation approach
  - References to similar features elsewhere

### Contributing Code

1. **Pick an issue** or create a new one
2. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make changes** following our coding standards
4. **Write tests** for new functionality
5. **Update documentation** as needed
6. **Commit changes**:
   ```bash
   git commit -m "Add: brief description of changes"
   ```
7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Create Pull Request**

## Adding New Problem Types

This is one of the most valuable contributions! Here's how:

### 1. Create Solver Class

Create `solvers/your_problem_solver.py`:

```python
from .base_solver import SpatialOptimizationProblem
from typing import Dict, Any, Optional
import geopandas as gpd

class YourProblemSolver(SpatialOptimizationProblem):
    """
    Your Problem Description
    
    Brief explanation of what this problem optimizes.
    """
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Full Problem Name",
            "short_name": "short-name",
            "category": "category",
            "description": "Clear, concise description",
            "mathematical_formulation": """
            Mathematical formulation in LaTeX or plain text
            """,
            "academic_refs": [
                "Citation 1 in APA format",
                "Citation 2 in APA format"
            ],
            "complexity": "Complexity class (e.g., NP-hard)",
            "typical_use_cases": [
                "Use case 1",
                "Use case 2"
            ],
            "keywords": [
                "keyword1", "keyword2"
            ],
            "variants": []
        }
    
    def get_conversation_prompts(self) -> Dict[str, Any]:
        return {
            "problem_detection": ["keyword1", "keyword2"],
            "parameter_questions": [
                {
                    "param": "param_name",
                    "question": "What is...?",
                    "type": "int|float|choice",
                    "validation": "Must be...",
                    "help": "This parameter controls..."
                }
            ],
            "constraint_suggestions": [],
            "explanation_template": "Template for explaining solutions"
        }
    
    def get_required_data(self) -> Dict[str, Dict[str, Any]]:
        return {
            "demand_points": {
                "required": True,
                "description": "What this data represents",
                "required_fields": [],
                "optional_fields": [],
                "geometry_type": "Point"
            }
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        # Validation logic
        if "required_param" not in params:
            return False, "Missing required parameter"
        return True, None
    
    def solve(
        self,
        data: Dict[str, gpd.GeoDataFrame],
        parameters: Dict[str, Any],
        constraints: Dict[str, Any],
        distance_metric: str = "euclidean"
    ) -> Dict[str, Any]:
        # Implementation
        pass
    
    def explain_solution(
        self,
        solution: Dict[str, Any],
        data: Dict[str, gpd.GeoDataFrame],
        detail_level: str = "standard"
    ) -> str:
        # Explanation logic
        pass
```

### 2. Register Problem

In `solvers/registry.py`, add to `_register_default_problems()`:

```python
def _register_default_problems(self):
    # ... existing registrations
    try:
        from .your_problem_solver import YourProblemSolver
        self.register(YourProblemSolver)
    except ImportError:
        pass
```

### 3. Write Tests

Create tests in `tests/test_solvers.py`:

```python
class TestYourProblemSolver(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.solver = problem_registry.get_problem("your-problem")
    
    def test_metadata(self):
        # Test metadata completeness
        pass
    
    def test_solve(self):
        # Test solving
        pass
    
    def test_validation(self):
        # Test parameter validation
        pass
```

### 4. Document

Create `docs/problems/your_problem.md`:

```markdown
# Your Problem Name

## Mathematical Formulation
[Detailed formulation]

## Complexity
[Analysis]

## Applications
[Real-world use cases]

## Academic References
[Full citations]

## Implementation Notes
[Algorithm details, assumptions]

## Usage Example
[Code example]
```

### 5. Checklist

- [ ] Solver class implements all required methods
- [ ] Metadata is complete and accurate
- [ ] Academic references in APA format
- [ ] Parameters have clear validation
- [ ] Solution includes all required fields
- [ ] Explanation works for all detail levels
- [ ] Tests pass
- [ ] Documentation is clear
- [ ] Example data/usage provided

## Coding Standards

### Python Style

Follow PEP 8 with these specifics:

- **Line length**: 100 characters max
- **Indentation**: 4 spaces
- **Imports**: Grouped (stdlib, third-party, local)
- **Docstrings**: Google style
- **Type hints**: Use for function signatures

### Formatting

Use `black` for formatting:
```bash
black solvers/ utils/ agent/ tests/
```

### Linting

Use `flake8`:
```bash
flake8 solvers/ utils/ agent/ --max-line-length=100
```

### Type Checking

Use `mypy` (optional but encouraged):
```bash
mypy solvers/ utils/ agent/
```

### Example Function

```python
def calculate_distance(
    origin: Point,
    destination: Point,
    metric: str = "euclidean"
) -> float:
    """
    Calculate distance between two points.
    
    Args:
        origin: Starting point
        destination: Ending point
        metric: Distance metric to use
        
    Returns:
        Distance value
        
    Raises:
        ValueError: If metric is invalid
    """
    if metric == "euclidean":
        return origin.distance(destination)
    raise ValueError(f"Unknown metric: {metric}")
```

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_solvers.py -v

# With coverage
pytest tests/ --cov=solvers --cov=utils --cov=agent

# Specific test
pytest tests/test_solvers.py::TestPMedianSolver::test_solve
```

### Writing Tests

- Test normal cases
- Test edge cases
- Test error conditions
- Use descriptive test names
- Include docstrings
- Use appropriate assertions

### Test Data

- Use synthetic data for unit tests
- Keep test data small and fast
- Document assumptions
- Store test data in `tests/test_data/`

## Documentation

### Code Documentation

- All public functions/classes need docstrings
- Use Google-style docstrings
- Include type hints
- Document parameters, returns, raises

### User Documentation

- Update README.md for user-facing changes
- Update architecture.md for design changes
- Create problem-specific docs in `docs/problems/`
- Include examples where helpful

### Academic Documentation

- Provide proper citations (APA format)
- Include mathematical formulations
- Document assumptions clearly
- Reference key papers

## Pull Request Process

1. **Before Creating PR**:
   - Ensure tests pass
   - Update documentation
   - Follow coding standards
   - Rebase on latest main

2. **PR Description**:
   - Clear title
   - What changes were made
   - Why these changes
   - Related issues (Fixes #123)
   - Testing performed
   - Screenshots if relevant

3. **Review Process**:
   - Address review comments
   - Keep discussion focused
   - Be open to suggestions
   - Update as needed

4. **After Merge**:
   - Delete your branch
   - Update your fork
   - Close related issues

## Questions?

- Open an issue for discussion
- Email maintainers
- Check existing documentation

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

---

Thank you for contributing to spatial optimization research! 🎯

