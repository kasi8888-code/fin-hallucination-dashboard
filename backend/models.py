from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from datetime import datetime

from database import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)

    prompt = Column(Text, nullable=False)

    response = Column(Text, nullable=False)

    overall_score = Column(Float, nullable=True)

    model = Column(String, nullable=True)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


class SentenceScore(Base):
    __tablename__ = "sentence_scores"

    id = Column(Integer, primary_key=True, index=True)

    analysis_id = Column(
        Integer,
        ForeignKey("analyses.id"),
        nullable=False
    )

    sentence = Column(Text, nullable=False)

    score = Column(Float, nullable=False)

    is_hallucinated = Column(
        Integer,
        nullable=False
    )

class LLMResponse(Base):
    __tablename__ = "llm_responses"

    id = Column(Integer, primary_key=True, index=True)

    analysis_id = Column(
        Integer,
        ForeignKey("analyses.id"),
        nullable=False
    )

    response_number = Column(
        Integer,
        nullable=False
    )

    response = Column(
        Text,
        nullable=False
    )