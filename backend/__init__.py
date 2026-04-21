"""SOCA FastAPI backend.

Thin REST + SSE layer over the existing Python modules (agent/, solvers/, utils/).
Nothing in this package reimplements optimisation or data-fetching logic —
each endpoint delegates to the same modules that the Streamlit app uses.
"""

__all__ = []
