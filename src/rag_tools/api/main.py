from __future__ import annotations

import re
from typing import List

from fastapi import FastAPI, HTTPException

from .schemas import (
    HealthResponse,
    LLMMetricsRequest,
    LLMMetricsResponse,
    RAGMetricsRequest,
    RAGMetricsResponse,
)
from .adapters import external_llm_metrics, external_rag_metrics

app = FastAPI(
    title="rag_tools API",
    version="0.1.0",
    description="Minimal API wrapper for RAG/LLM metrics (software engineering part).",
)

_word_re = re.compile(r"[A-Za-zА-Яа-я0-9_]+", re.UNICODE)


def _tokens(text: str) -> List[str]:
    return _word_re.findall((text or "").lower())


def _token_f1(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0

    from collections import Counter

    ca, cb = Counter(ta), Counter(tb)
    common = sum((ca & cb).values())
    precision = common / max(len(tb), 1)
    recall = common / max(len(ta), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _hit_at_k(relevant_ids: List[int], predicted_ids: List[int], k: int) -> int:
    topk = predicted_ids[:k]
    rel = set(relevant_ids)
    return 1 if any(pid in rel for pid in topk) else 0


def _mrr(relevant_ids: List[int], predicted_ids: List[int]) -> float:
    rel = set(relevant_ids)
    for rank, pid in enumerate(predicted_ids, start=1):
        if pid in rel:
            return 1.0 / rank
    return 0.0


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/metrics/llm", response_model=LLMMetricsResponse)
def llm_metrics(req: LLMMetricsRequest) -> LLMMetricsResponse:
    # baseline metrics ثابتة (علشان tests تفضل ثابتة)
    baseline = {
        "exact_match": (req.reference.strip() == req.candidate.strip()),
        "token_f1": round(_token_f1(req.reference, req.candidate), 4),
        "len_reference": len(req.reference),
        "len_candidate": len(req.candidate),
    }

    ext = external_llm_metrics(req.reference, req.candidate)
    if ext is None:
        return LLMMetricsResponse(metrics=baseline)

    merged = {**baseline, **ext}
    return LLMMetricsResponse(metrics=merged)


@app.post("/metrics/rag", response_model=RAGMetricsResponse)
def rag_metrics(req: RAGMetricsRequest) -> RAGMetricsResponse:
    if len(req.relevant) != len(req.predicted):
        raise HTTPException(status_code=400, detail="relevant and predicted must have the same length")

    hits = []
    mrrs = []
    for rel, pred in zip(req.relevant, req.predicted):
        hits.append(_hit_at_k(rel, pred, req.k))
        mrrs.append(_mrr(rel, pred))

    baseline = {
        f"hit@{req.k}": round(sum(hits) / max(len(hits), 1), 4),
        "mrr": round(sum(mrrs) / max(len(mrrs), 1), 4),
        "n_questions": len(hits),
    }

    ext = external_rag_metrics(req.relevant, req.predicted, req.k)
    if ext is None:
        return RAGMetricsResponse(k=req.k, metrics=baseline)

    merged = {**baseline, **ext}
    return RAGMetricsResponse(k=req.k, metrics=merged)