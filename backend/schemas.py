from pydantic import BaseModel
from datetime import datetime


class AnalysisRequest(BaseModel):
    prompt: str
    # Optional trusted-source document used for groundedness scoring. When
    # present, the brain compares every sampled answer against it (this is the
    # dominant hallucination signal). When absent, only self-consistency is used.
    reference: str | None = None


class SentenceScoreResponse(BaseModel):
    sentence: str
    score: float
    is_hallucinated: int


class LLMResponseItem(BaseModel):
    response_number: int
    response: str


class AnalysisResponse(BaseModel):
    id: int
    prompt: str
    response: str
    overall_score: float
    model: str
    # Calibrated verdict from the brain (Faithful | Suspicious | Hallucinated).
    decision: str | None = None
    sentence_scores: list[SentenceScoreResponse]


class HistoryItem(BaseModel):
    id: int
    prompt: str
    response: str
    overall_score: float
    model: str
    decision: str | None = None
    created_at: datetime


class DetailedAnalysisResponse(BaseModel):
    id: int
    prompt: str
    response: str
    overall_score: float
    model: str
    decision: str | None = None
    created_at: datetime
    llm_responses: list[LLMResponseItem]
    sentence_scores: list[SentenceScoreResponse]