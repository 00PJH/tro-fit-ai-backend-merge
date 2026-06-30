import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.db.session import engine
from app.db.base import Base
from app.api.v1.routers import api_router
from app.models import user  # noqa: F401
from app.models.rom import RomHistory  # noqa: F401

# 앱 시작 시 DB 테이블 초기화 및 생성
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Tro-Fit Auth System (JWT)")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API v1 라우터 등록
app.include_router(api_router, prefix="/api/v1")

# 정적 HTML 파일 경로 설정
STATIC_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")

@app.get("/")
def read_root():
    """index.html 파일 서빙"""
    if not os.path.exists(STATIC_HTML_PATH):
        raise HTTPException(status_code=404, detail="index.html 파일을 찾을 수 없습니다.")
    return FileResponse(STATIC_HTML_PATH)

