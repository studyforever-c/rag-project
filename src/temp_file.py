from pathlib import Path
from fastapi import UploadFile
import tempfile

async def temp_file(file: UploadFile) -> Path:
    """为用户上传文件设置临时文件"""
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete = False, suffix = suffix) as tmp:
        content = await file.read()
        tmp.write(content)
    return Path(tmp.name)