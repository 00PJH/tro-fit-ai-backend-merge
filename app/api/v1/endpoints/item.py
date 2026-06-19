from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from app.db.session import get_db
from app.schemas import item as item_schemas
from app.crud import item as item_crud
from app.api.v1.endpoints.user import get_current_user
from app.models.user import User

router = APIRouter()

@router.get("/", response_model=List[item_schemas.Item])
def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return item_crud.get_items(db, skip=skip, limit=limit)

@router.post("/", response_model=item_schemas.Item)
def create_item_for_user(
    item: item_schemas.ItemCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return item_crud.create_user_item(db=db, item=item, user_id=current_user.user_id)
