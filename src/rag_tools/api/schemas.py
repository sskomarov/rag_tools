from pydantic import BaseModel, Field
from typing import Any, Dict, List


class HealthResponse(BaseModel):
    status: str = "ok"


class LLMMetricsRequest(BaseModel):
    reference: str = Field(..., description="Reference (gold) answer text")
    candidate: str = Field(..., description="Model generated answer text")


class LLMMetricsResponse(BaseModel):
    metrics: Dict[str, Any]


class RAGMetricsRequest(BaseModel):
    relevant: List[List[int]] = Field(
        ...,
        description="Per-question list of relevant ids"
    )
    predicted: List[List[int]] = Field(
        ...,
        description="Per-question ranked list of predicted ids"
    )
    k: int = Field(5, ge=1, le=100)


class RAGMetricsResponse(BaseModel):
    k: int
    metrics: Dict[str, Any]
