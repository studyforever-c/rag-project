from fastapi import FastAPI
from routers.local_base import router as local_base
from routers.input_base import router as input_base
app = FastAPI(title='Rag实战')

app.include_router(local_base, prefix = '/local_base')
app.include_router(input_base, prefix = '/input_base')