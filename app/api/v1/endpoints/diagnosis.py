from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.diagnosis import DiagnosisMetric, DiagnosisSession
from app.models.user import User
from app.schemas.diagnosis import DiagnosisCreate, DiagnosisResponse


router = APIRouter()


@router.post("/", response_model=DiagnosisResponse, status_code=status.HTTP_201_CREATED)
def create_diagnosis(payload: DiagnosisCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.user_id == payload.user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    session = DiagnosisSession(
        user_id=payload.user_id,
        source=payload.source,
        landmark_file_path=payload.landmark_file_path,
        total_frames=payload.summary.total_frames,
        detected_frames=payload.summary.detected_frames,
        detection_rate=payload.summary.detection_rate,
    )
    db.add(session)
    db.flush()

    metric = DiagnosisMetric(
        session_id=session.id,
        knee_left_rom=payload.metrics.knee_left_rom,
        knee_right_rom=payload.metrics.knee_right_rom,
        shoulder_left_rom=payload.metrics.shoulder_left_rom,
        shoulder_right_rom=payload.metrics.shoulder_right_rom,
        spine_flexion_rom=payload.metrics.spine_flexion_rom,
        knee_asymmetry_detected=payload.diagnosis.knee_asymmetry_detected,
        shoulder_asymmetry_detected=payload.diagnosis.shoulder_asymmetry_detected,
        falls_risk_score=payload.diagnosis.falls_risk_score,
    )
    db.add(metric)
    db.commit()
    db.refresh(session)

    return DiagnosisResponse(
        session_id=session.id,
        user_id=session.user_id,
        source=session.source,
        landmark_file_path=session.landmark_file_path,
        summary=payload.summary,
        metrics=payload.metrics,
        diagnosis=payload.diagnosis,
    )


@router.get("/{session_id}", response_model=DiagnosisResponse)
def read_diagnosis(session_id: int, db: Session = Depends(get_db)):
    session = db.query(DiagnosisSession).filter(DiagnosisSession.id == session_id).first()
    if session is None or session.metrics is None:
        raise HTTPException(status_code=404, detail="Diagnosis session not found")

    return DiagnosisResponse(
        session_id=session.id,
        user_id=session.user_id,
        source=session.source,
        landmark_file_path=session.landmark_file_path,
        summary={
            "total_frames": session.total_frames,
            "detected_frames": session.detected_frames,
            "detection_rate": session.detection_rate,
        },
        metrics={
            "knee_left_rom": session.metrics.knee_left_rom,
            "knee_right_rom": session.metrics.knee_right_rom,
            "shoulder_left_rom": session.metrics.shoulder_left_rom,
            "shoulder_right_rom": session.metrics.shoulder_right_rom,
            "spine_flexion_rom": session.metrics.spine_flexion_rom,
        },
        diagnosis={
            "knee_asymmetry_detected": session.metrics.knee_asymmetry_detected,
            "shoulder_asymmetry_detected": session.metrics.shoulder_asymmetry_detected,
            "falls_risk_score": session.metrics.falls_risk_score,
        },
    )
