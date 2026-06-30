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
    measurement_type: str
    joint: str
    movement: str
    measurement: Optional[Any] = None
    frames: List[FrameData]


# ==============================================================================
# Response Schemas
# ==============================================================================

class RomResultItem(BaseModel):
    neutral_angle: Optional[float] = None
    max_angle: Optional[float] = None
    rom: Optional[float] = None
    reliable: bool
    reason: Optional[str] = None

class RomRatioItem(BaseModel):
    rom_deg: float
    normal_deg: float
    rom_ratio_pct: float
    grade: str

class MeasurementMeta(BaseModel):
    neutral_captured_at: str
    max_candidates: List[str]
    max_selected: str
    selection_reason: str
    use_world_landmarks: bool
    visibility_threshold: float

class RomAnalyzeResponse(BaseModel):
    joint: str
    movement: str
    measurement: MeasurementMeta
    rom_results: Dict[str, RomResultItem] = Field(
        ..., description="Dict mapping joints (e.g. left_shoulder) to ROM calculation results"
    )
    rom_ratio: Dict[str, RomRatioItem] = Field(
        ..., description="Dict mapping joints to ROM achievement ratios and grades"
    )
    confidence: str
    elapsed_sec: float
    history_id: Optional[int] = Field(None, description="Database ID of the saved measurement record")
    created_at: Optional[str] = Field(None, description="ISO timestamp of when the record was created")
