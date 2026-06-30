from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, func

from app.db.base import Base


class RomHistory(Base):
    __tablename__ = "rom_histories"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=True, index=True)
    session_id = Column(String, nullable=False, index=True)
    measurement_type = Column(String, nullable=False)
    joint = Column(String, nullable=False)
    movement = Column(String, nullable=False)
    rom_results = Column(JSON, nullable=False)
    rom_ratio = Column(JSON, nullable=False)
    measurement_meta = Column(JSON, nullable=False)
    confidence = Column(String, nullable=False)
    elapsed_sec = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
