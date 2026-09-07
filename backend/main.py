from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from database import engine, Base, SessionLocal
from models import Analysis, SentenceScore, LLMResponse
from schemas import (
    AnalysisRequest,
    AnalysisResponse,
    HistoryItem,
    DetailedAnalysisResponse
)

from llm import DEFAULT_MODEL
from scoring import (
    generate_multiple_answers,
    score_responses
)


Base.metadata.create_all(bind=engine)

app = FastAPI(title="Hallucination Risk API")


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


@app.get("/")
def home():
    return {
        "message": "Hallucination Risk API is running"
    }


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(
    request: AnalysisRequest,
    db: Session = Depends(get_db)
):

    # Generate multiple independent answers. SelfCheckGPT requires varied
    # stochastic samples, so a sampling temperature is applied (the brain's
    # consistency signal is meaningless if all answers are identical).
    try:

        answers = generate_multiple_answers(
            request.prompt,
            number_of_answers=3,
            temperature=0.8
        )

    except Exception as e:

        print("LLM ERROR:", repr(e))

        raise HTTPException(
            status_code=503,
            detail=f"LLM API error: {str(e)}"
        )

    try:

        # First answer is used as the displayed response
        response_text = answers[0]

        # Score through the brain: groundedness vs an optional trusted
        # reference (dominant signal) fused with self-consistency stats.
        result = score_responses(
            question=request.prompt,
            answers=answers,
            reference=request.reference,
            primary=response_text
        )

        risk = float(result["risk_score"])
        decision = str(result["decision"])
        sentence_scores = [
            {
                "sentence": item["sentence"],
                "score": float(item["risk"]),
                "is_hallucinated": int(item["risk"] >= 0.5)
            }
            for item in result["sentence_risks"]
        ]

        # Create main analysis
        analysis = Analysis(
            prompt=request.prompt,
            response=response_text,
            overall_score=risk,
            decision=decision,
            model=DEFAULT_MODEL
        )

        db.add(analysis)

        # Flush so analysis.id becomes available
        db.flush()

        # Save all sampled responses
        for i, answer in enumerate(answers, start=1):

            llm_response = LLMResponse(
                analysis_id=analysis.id,
                response_number=i,
                response=answer
            )

            db.add(llm_response)

        # Save sentence-level scores
        for result in sentence_scores:

            sentence_score = SentenceScore(
                analysis_id=analysis.id,
                sentence=result["sentence"],
                score=result["score"],
                is_hallucinated=result["is_hallucinated"]
            )

            db.add(sentence_score)

        # Commit everything together
        db.commit()

        # Refresh analysis after commit
        db.refresh(analysis)

    except Exception:

        # Undo all database changes if anything fails
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="Failed to save analysis."
        )

    return {
        "id": analysis.id,
        "prompt": analysis.prompt,
        "response": analysis.response,
        "overall_score": analysis.overall_score,
        "model": analysis.model,
        "decision": analysis.decision,
        "sentence_scores": sentence_scores
    }

@app.get("/history", response_model=list[HistoryItem])
def get_history(
    db: Session = Depends(get_db)
):

    analyses = db.query(Analysis).order_by(
        Analysis.id.desc()
    ).all()

    return [
        {
            "id": analysis.id,
            "prompt": analysis.prompt,
            "response": analysis.response,
            "overall_score": analysis.overall_score,
            "model": analysis.model,
            "decision": analysis.decision,
            "created_at": analysis.created_at
        }
        for analysis in analyses
    ]

@app.get(
    "/analysis/{analysis_id}",
    response_model=DetailedAnalysisResponse
)
def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db)
):

    analysis = db.query(Analysis).filter(
        Analysis.id == analysis_id
    ).first()

    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )

    llm_responses = db.query(
        LLMResponse
    ).filter(
        LLMResponse.analysis_id == analysis_id
    ).order_by(
        LLMResponse.response_number
    ).all()

    sentence_scores = db.query(
        SentenceScore
    ).filter(
        SentenceScore.analysis_id == analysis_id
    ).all()

    return {
        "id": analysis.id,
        "prompt": analysis.prompt,
        "response": analysis.response,
        "overall_score": analysis.overall_score,
        "model": analysis.model,
        "decision": analysis.decision,
        "created_at": analysis.created_at,

        "llm_responses": [
            {
                "response_number": response.response_number,
                "response": response.response
            }
            for response in llm_responses
        ],

        "sentence_scores": [
            {
                "sentence": score.sentence,
                "score": score.score,
                "is_hallucinated": score.is_hallucinated
            }
            for score in sentence_scores
        ]
    }