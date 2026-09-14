from fastapi import UploadFile, File
from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str
    top_k: int = 15

class AskResponse(BaseModel):
    history: list[dict]