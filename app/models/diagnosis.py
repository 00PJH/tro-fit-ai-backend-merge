from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship
from app.db.base import Base


class DiagnosisSession(Base):
    __tablename__ = "diagnosis_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.user_id"), index=True, nullable=False)
    source = Column(String, nullable=False, default="webcam")
    landmark_file_path = Column(String, nullable=True)
    total_frames = Column(Integer, nullable=False, default=0)
    detected_frames = Column(Integer, nullable=False, default=0)
    detection_rate = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, server_default=func.now())

    metrics = relationship(
        "DiagnosisMetric",
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
    )


class DiagnosisMetric(Base):
    __tablename__ = "diagnosis_metrics"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("diagnosis_sessions.id"), unique=True, nullable=False)
    knee_left_rom = Column(Float, nullable=False, default=0.0)
    knee_right_rom = Column(Float, nullable=False, default=0.0)
    shoulder_left_rom = Column(Float, nullable=False, default=0.0)
    shoulder_right_rom = Column(Float, nullable=False, default=0.0)
    spine_flexion_rom = Column(Float, nullable=False, default=0.0)
    knee_asymmetry_detected = Column(Boolean, nullable=False, default=False)
    shoulder_asymmetry_detected = Column(Boolean, nullable=False, default=False)
    falls_risk_score = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, server_default=func.now())

    session = relationship("DiagnosisSession", back_populates="metrics")
