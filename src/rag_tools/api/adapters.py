from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Dict, Iterable, Optional


def _try_import_callable(spec: str) -> Optional[Callable[..., Any]]:
    # spec format: "module.sub:func"
    if ":" not in spec:
        return None
    mod_name, fn_name = spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        if callable(fn):
            return fn
    except Exception:
        return None
    return None


def _resolve_callable(
        env_var: str,
        candidates: Iterable[str],
) -> Optional[Callable[..., Any]]:
    # 1) env override
    env_spec = os.getenv(env_var, "").strip()
    if env_spec:
        fn = _try_import_callable(env_spec)
        if fn:
            return fn

    # 2) common candidate locations
    for spec in candidates:
        fn = _try_import_callable(spec)
        if fn:
            return fn
    return None


# ✅ "العقد" اللي سيرجي يلتزم بيه (أو يغيره عبر env vars)
_LLM_CANDIDATES = [
    "rag_tools.metrics.llm:compute_llm_metrics",
    "rag_tools.metrics.llm:calculate_all",
    "rag_tools.metrics.llm_metrics:compute_llm_metrics",
    "rag_tools.metrics.llm_metrics:calculate_all",
]

_RAG_CANDIDATES = [
    "rag_tools.metrics.rag:compute_rag_metrics",
    "rag_tools.metrics.rag:evaluate_retrieval",
    "rag_tools.metrics.rag_metrics:compute_rag_metrics",
    "rag_tools.metrics.rag_metrics:evaluate_retrieval",
]


def external_llm_metrics(reference: str, candidate: str) -> Optional[Dict[str, Any]]:
    fn = _resolve_callable("RAGTOOLS_LLM_METRICS_CALLABLE", _LLM_CANDIDATES)
    if not fn:
        return None

    out = fn(reference=reference, candidate=candidate)  # preferred signature

    if isinstance(out, dict):
        return out
    if hasattr(out, "model_dump"):
        return out.model_dump()
    if hasattr(out, "dict"):
        return out.dict()
    return {"result": out}


def external_rag_metrics(
        relevant: list[list[int]],
        predicted: list[list[int]],
        k: int,
) -> Optional[Dict[str, Any]]:
    fn = _resolve_callable("RAGTOOLS_RAG_METRICS_CALLABLE", _RAG_CANDIDATES)
    if not fn:
        return None

    out = fn(relevant=relevant, predicted=predicted, k=k)  # preferred signature

    if isinstance(out, dict):
        return out
    if hasattr(out, "model_dump"):
        return out.model_dump()
    if hasattr(out, "dict"):
        return out.dict()
    return {"result": out}
