import jwt
from typing import Optional
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.core.config import settings
from app.crud import user as user_crud
from app.db.session import get_db
from app.models.rom import RomHistory
from app.models.user import User
from app.schemas.rom import RomAnalyzeRequest, RomAnalyzeResponse, RomHistoryListResponse
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


def format_datetime_to_kst_iso(dt: datetime) -> str:
    kst_tz = timezone(timedelta(hours=9))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=kst_tz)
    else:
        dt = dt.astimezone(kst_tz)
    return dt.isoformat()


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
        
        # 2. Dynamic week_number calculation based on first measurement
        now_time = datetime.now()
        week_number = 1
        if current_user:
            first_record = db.query(RomHistory).filter(
                RomHistory.user_id == current_user.user_id
            ).order_by(RomHistory.measured_at.asc()).first()
            
            if first_record:
                first_date = first_record.measured_at
                if first_date.tzinfo is not None:
                    first_date = first_date.replace(tzinfo=None)
                days_diff = (now_time - first_date).days
                week_number = (days_diff // 7) + 1
        
        # 3. Save result to database
        db_history = RomHistory(
            user_id=current_user.user_id if current_user else None,
            session_id=request.session_id,
            measurement_type=request.measurement_type or results.get("side", "both"),
            joint=request.joint,
            movement=request.movement,
            side=results.get("side", "both"),
            video_file=results.get("video_file"),
            video_info=results.get("video_info"),
            rom_results=results.get("rom_results", {}),
            mobility_analysis=results.get("mobility_analysis", []),
            measurement=results.get("measurement", {}),
            confidence=results.get("confidence", "LOW"),
            elapsed_sec=results.get("elapsed_sec", 0.0),
            model=results.get("model"),
            week_number=week_number,
            measured_at=now_time
        )
        db.add(db_history)
        db.commit()
        db.refresh(db_history)
        
        # 4. Fill output values for RomAnalyzeResponse schema
        results["week_number"] = db_history.week_number
        results["measured_at"] = format_datetime_to_kst_iso(db_history.measured_at)
        
        return results
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err)
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ROM 분석 도중 내부 서버 에러가 발생했습니다: {str(exc)}"
        )


@router.get("/rom/history", response_model=RomHistoryListResponse, status_code=status.HTTP_200_OK)
def get_rom_history(
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    GET /api/v1/analyze/rom/history
    
    Returns logged-in user's measurement records ordered by week_number (measured_at).
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="로그인이 필요한 기능입니다."
        )
    
    records = db.query(RomHistory).filter(
        RomHistory.user_id == current_user.user_id
    ).order_by(RomHistory.week_number.asc(), RomHistory.measured_at.asc()).all()
    
    results = []
    for r in records:
        results.append({
            "session_id": r.session_id,
            "week_number": r.week_number or 1,
            "measured_at": format_datetime_to_kst_iso(r.measured_at),
            "joint": r.joint,
            "movement": r.movement,
            "side": r.side or "both",
            "mobility_analysis": r.mobility_analysis or [],
            "confidence": r.confidence
        })
        
    return {"results": results}
