import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from app.db.session import engine
from app.db.base import Base
from app.api.v1.routers import api_router
from app.models import user, item, diagnosis  # noqa: F401

# ???쒖옉 ??DB ?뚯씠釉?珥덇린??諛??앹꽦
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Tro-Fit Auth System (JWT)")

# API v1 ?쇱슦???깅줉
app.include_router(api_router, prefix="/api/v1")

# ?뺤쟻 HTML ?뚯씪 寃쎈줈 ?ㅼ젙
STATIC_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")

@app.get("/")
def read_root():
    """index.html ?뚯씪 ?쒕튃"""
    if not os.path.exists(STATIC_HTML_PATH):
        raise HTTPException(status_code=404, detail="index.html ?뚯씪??李얠쓣 ???놁뒿?덈떎.")
    return FileResponse(STATIC_HTML_PATH)

