from pydantic import BaseModel
from datetime import datetime


class AnalysisRequest(BaseModel):
    prompt: str


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
    sentence_scores: list[SentenceScoreResponse]


class HistoryItem(BaseModel):
    id: int
    prompt: str
    response: str
    overall_score: float
    model: str
    created_at: datetime


class DetailedAnalysisResponse(BaseModel):
    id: int
    prompt: str
    response: str
    overall_score: float
    model: str
    created_at: datetime
    llm_responses: list[LLMResponseItem]
    sentence_scores: list[SentenceScoreResponse]