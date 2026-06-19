import os
import secrets
import hashlib
import jwt
from datetime import datetime, timedelta
from typing import Union, Any
from app.core.config import settings

def hash_password(password: str) -> tuple[str, str]:
    """임의의 솔트(salt)를 생성하여 비밀번호 해싱"""
    salt = secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac(
        hash_name="sha256",
        password=password.encode("utf-8"),
        salt=salt.encode("utf-8"),
        iterations=100000
    ).hex()
    return pw_hash, salt

def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    """솔트를 사용하여 입력받은 비밀번호 검증"""
    pw_hash = hashlib.pbkdf2_hmac(
        hash_name="sha256",
        password=password.encode("utf-8"),
        salt=salt.encode("utf-8"),
        iterations=100000
    ).hex()
    return secrets.compare_digest(pw_hash, stored_hash)

def create_access_token(subject: Union[str, Any], expires_delta: timedelta = None) -> str:
    """JWT Access Token 생성"""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt
