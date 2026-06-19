from fastapi import APIRouter
from app.api.v1.endpoints import user, item, diagnosis

api_router = APIRouter()
api_router.include_router(user.router, prefix="", tags=["auth"])
api_router.include_router(item.router, prefix="/items", tags=["items"])
api_router.include_router(diagnosis.router, prefix="/diagnosis", tags=["diagnosis"])
