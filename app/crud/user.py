from sqlalchemy.orm import Session
from app.models.user import User
from app.schemas.user import UserSignup
from app.core.security import hash_password

def get_user_by_id(db: Session, user_id: str) -> User | None:
    """ID로 사용자 조회"""
    return db.query(User).filter(User.user_id == user_id).first()

def get_user_by_email(db: Session, email: str) -> User | None:
    """이메일로 사용자 조회"""
    return db.query(User).filter(User.user_email == email).first()

def create_user(db: Session, user_in: UserSignup) -> User:
    """새 사용자 생성"""
    pw_hash, salt = hash_password(user_in.user_password)
    db_user = User(
        user_id=user_in.user_id,
        password_hash=pw_hash,
        salt=salt,
        user_email=user_in.user_email,
        user_name=user_in.user_name,
        user_age=user_in.user_age,
        user_height=user_in.user_height,
        user_weight=user_in.user_weight
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
