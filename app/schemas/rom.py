from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# ==============================================================================
# Request Schemas
# ==============================================================================

class Landmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: Optional[float] = 1.0
    pixel_x: Optional[float] = None
    pixel_y: Optional[float] = None

class WorldLandmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: Optional[float] = 1.0

class PoseData(BaseModel):
    pose_index: int
    landmarks: Dict[str, Landmark] = Field(
        ..., description="Dict mapping landmark names (e.g. left_shoulder) to coordinates"
    )
    world_landmarks: Dict[str, WorldLandmark] = Field(
        ..., description="Dict mapping landmark names to world coordinates"
    )

class FrameData(BaseModel):
    frame_index: int
    timestamp_ms: int
    detected: bool
    num_poses: int
    image_width: int
    image_height: int
    poses: List[PoseData]

class RomAnalyzeRequest(BaseModel):
    session_id: str
    measurement_type: Optional[str] = None
    joint: str
    movement: str
    side: Optional[str] = None
    video_file: Optional[str] = None
    video_info: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    measurement: Optional[Any] = None
    frames: List[FrameData]


# ==============================================================================
# Response Schemas
# ==============================================================================

class RomResultItem(BaseModel):
    neutral_angle: Optional[float] = None
    max_angle: Optional[float] = None
    ama_neutral_angle: Optional[float] = None
    ama_max_angle: Optional[float] = None
    rom: Optional[float] = None
    reliable: bool
    reason: Optional[str] = None

class MobilityScore(BaseModel):
    normal_deg: float
    rom_ratio: float
    grade: str
    clinical_meaning: str

class MobilityAnalysisItem(BaseModel):
    side: str
    measured_angle_deg: float
    mobility_score: MobilityScore

class MeasurementMeta(BaseModel):
    neutral_captured_at: str
    max_candidates: List[str]
    max_selected: str
    selection_reason: str
    use_world_landmarks: bool
    visibility_threshold: float

class RomAnalyzeResponse(BaseModel):
    session_id: str
    week_number: int
    measured_at: str
    joint: str
    movement: str
    side: str
    mobility_analysis: List[MobilityAnalysisItem] = Field(
        ..., description="List of mobility scores and grades per side"
    )
    confidence: str

class RomHistoryItem(BaseModel):
    session_id: str
    week_number: int
    measured_at: str
    joint: str
    movement: str
    side: str
    mobility_analysis: List[MobilityAnalysisItem] = Field(
        ..., description="List of mobility scores and grades per side"
    )
    confidence: str

class RomHistoryListResponse(BaseModel):
    results: List[RomHistoryItem]
