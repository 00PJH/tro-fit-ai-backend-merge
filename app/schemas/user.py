from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional

class UserSignup(BaseModel):
    user_id: str = Field(..., min_length=3, max_length=50, description="사용자 아이디 (영문/숫자)")
    user_password: str = Field(..., min_length=4, description="비밀번호")
    user_email: EmailStr = Field(..., description="이메일 주소")
    user_name: str = Field(..., min_length=1, description="사용자 실명")
    user_age: int = Field(..., ge=1, le=150, description="나이")
    user_height: float = Field(..., ge=30.0, le=300.0, description="키 (cm)")
    user_weight: float = Field(..., ge=10.0, le=500.0, description="몸무게 (kg)")

class UserLogin(BaseModel):
    user_id: str
    user_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserProfileResponse(BaseModel):
    user_id: str
    user_email: str
    user_name: str
    user_age: int
    user_height: float
    user_weight: float

    class Config:
        from_attributes = True

