"""Expose the problem registry so the React UI can render parameter forms."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter

from solvers.registry import problem_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/problems", tags=["problems"])


def _problem_payload(meta: Dict[str, Any]) -> Dict[str, Any]:
    short_name = meta.get("short_name")
    problem = problem_registry.get_problem(short_name) if short_name else None
    prompts: Dict[str, Any] = {}
    viz: Dict[str, Any] = {}
    if problem is not None:
        try:
            prompts = problem.get_conversation_prompts() or {}
        except Exception:
            prompts = {}
        try:
            viz = problem.get_visualization_config() or {}
        except Exception:
            viz = {}
    return {
        "short_name": short_name,
        "name": meta.get("name"),
        "category": meta.get("category"),
        "description": meta.get("description"),
        "keywords": meta.get("keywords", []),
        "variants": meta.get("variants", []),
        "complexity": meta.get("complexity"),
        "typical_use_cases": meta.get("typical_use_cases", []),
        "conversation_prompts": prompts,
        "visualization_config": viz,
    }


@router.get("")
def list_problems() -> Dict[str, List[Dict[str, Any]]]:
    payloads = [_problem_payload(m) for m in problem_registry.list_problems()]
    return {"problems": payloads}
