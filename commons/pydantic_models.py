from fastapi import UploadFile, File
from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    history: list[dict]