import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.core.config import settings
from app.core.security import verify_password, create_access_token
from app.schemas.user import UserSignup, UserLogin, UserProfileResponse, Token
from app.crud import user as user_crud
from app.models.user import User

router = APIRouter()
security = HTTPBearer()

def get_current_user(
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """HTTP Authorization 헤더의 Bearer 토큰 검증 후 현재 사용자 객체 반환"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="유효하지 않은 인증 토큰입니다."
            )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증 토큰이 만료되었습니다. 다시 로그인해주세요."
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="올바르지 않은 인증 토큰입니다."
        )
        
    db_user = user_crud.get_user_by_id(db, user_id=user_id)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="사용자를 찾을 수 없습니다."
        )
    return db_user

@router.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(user_data: UserSignup, db: Session = Depends(get_db)):
    """회원가입 API"""
    # 1. 아이디 중복 검사
    if user_crud.get_user_by_id(db, user_id=user_data.user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 존재하는 아이디입니다."
        )
        
    # 2. 이메일 중복 검사
    if user_crud.get_user_by_email(db, email=user_data.user_email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 사용 중인 이메일입니다."
        )
        
    # 3. DB 저장
    user_crud.create_user(db, user_in=user_data)
    return {"message": "회원가입이 정상적으로 처리되었습니다."}

@router.post("/login", response_model=Token)
def login(login_data: UserLogin, db: Session = Depends(get_db)):
    """로그인 API (성공 시 access token 발급)"""
    db_user = user_crud.get_user_by_id(db, user_id=login_data.user_id)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="아이디 또는 비밀번호가 올바르지 않습니다."
        )
        
    # 비밀번호 대조 검증
    if not verify_password(login_data.user_password, db_user.salt, db_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="아이디 또는 비밀번호가 올바르지 않습니다."
        )
        
    # JWT 토큰 생성
    token = create_access_token(subject=db_user.user_id)
    return {"access_token": token, "token_type": "bearer"}

@router.post("/logout")
def logout():
    """로그아웃 API (stateless JWT이므로 클라이언트측에서 토큰을 폐기하며, 서버는 성공 메시지만 반환)"""
    return {"message": "로그아웃 되었습니다."}

@router.get("/me", response_model=UserProfileResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """현재 로그인된 유저 조회 API"""
    return current_user
