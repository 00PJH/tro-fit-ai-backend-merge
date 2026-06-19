from pydantic import BaseModel, Field
from typing import Optional


class DiagnosisSummary(BaseModel):
    total_frames: int = Field(default=0, ge=0)
    detected_frames: int = Field(default=0, ge=0)
    detection_rate: float = Field(default=0.0, ge=0.0, le=100.0)


class DiagnosisMetrics(BaseModel):
    knee_left_rom: float = 0.0
    knee_right_rom: float = 0.0
    shoulder_left_rom: float = 0.0
    shoulder_right_rom: float = 0.0
    spine_flexion_rom: float = 0.0


class DiagnosisFlags(BaseModel):
    knee_asymmetry_detected: bool = False
    shoulder_asymmetry_detected: bool = False
    falls_risk_score: float = Field(default=0.0, ge=0.0)


class DiagnosisCreate(BaseModel):
    user_id: str
    source: str = "webcam"
    landmark_file_path: Optional[str] = None
    summary: DiagnosisSummary
    metrics: DiagnosisMetrics
    diagnosis: DiagnosisFlags


class DiagnosisResponse(BaseModel):
    session_id: int
    user_id: str
    source: str
    landmark_file_path: Optional[str]
    summary: DiagnosisSummary
    metrics: DiagnosisMetrics
    diagnosis: DiagnosisFlags

    class Config:
        from_attributes = True
