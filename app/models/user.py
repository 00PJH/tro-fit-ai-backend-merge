from sqlalchemy import Column, String, Integer, Float, DateTime, func
from app.db.base import Base

class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, index=True)
    password_hash = Column(String, nullable=False)
    salt = Column(String, nullable=False)
    user_email = Column(String, unique=True, index=True, nullable=False)
    user_name = Column(String, nullable=False)
    user_age = Column(Integer, nullable=False)
    user_height = Column(Float, nullable=False)
    user_weight = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
