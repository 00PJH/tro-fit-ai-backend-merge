from fastapi import APIRouter
from app.api.v1.endpoints import user, rom

api_router = APIRouter()
api_router.include_router(user.router, prefix="", tags=["auth"])
api_router.include_router(rom.router, prefix="/analyze", tags=["analyze"])
