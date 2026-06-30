import jwt
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.core.config import settings
from app.crud import user as user_crud
from app.db.session import get_db
from app.models.rom import RomHistory
from app.models.user import User
from app.schemas.rom import RomAnalyzeRequest, RomAnalyzeResponse
from app.services.rom_analysis import analyze_rom_from_frames

router = APIRouter()

bearer_opt = HTTPBearer(auto_error=False)

def get_current_user_optional(
    db: Session = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_opt),
) -> Optional[User]:
    if not credentials:
        return None
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        return user_crud.get_user_by_id(db, user_id=user_id)
    except jwt.PyJWTError:
        return None


@router.post("/rom", response_model=RomAnalyzeResponse, status_code=status.HTTP_200_OK)
def analyze_rom(
    request: RomAnalyzeRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    POST /api/v1/analyze/rom
    
    Accepts on-device extracted pose landmarks, calculates joint Range of Motion (ROM),
    and saves the measurement record to the database (linked to user if authenticated).
    """
    try:
        # 1. Perform ROM Analysis
        results = analyze_rom_from_frames(request)
        
        # 2. Save result to database
        db_history = RomHistory(
            user_id=current_user.user_id if current_user else None,
            session_id=request.session_id,
            measurement_type=request.measurement_type,
            joint=request.joint,
            movement=request.movement,
            rom_results=results.get("rom_results", {}),
            rom_ratio=results.get("rom_ratio", {}),
            measurement_meta=results.get("measurement", {}),
            confidence=results.get("confidence", "LOW"),
            elapsed_sec=results.get("elapsed_sec", 0.0),
        )
        db.add(db_history)
        db.commit()
        db.refresh(db_history)
        
        # 3. Add database metadata to response payload
        results["history_id"] = db_history.id
        results["created_at"] = db_history.created_at.isoformat()
        
        return results
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ROM 분석 도중 내부 서버 에러가 발생했습니다: {str(exc)}"
        )
