import numpy as np
import faiss
import json
from fastapi import FastAPI
from pydantic import BaseModel
from src.embed import EmbeddingModel
from src.llm import llm
app = FastAPI(title = '基于RAG的本地知识库问答助手')

class AskRequest(BaseModel):
    question: str
    top_k: int = 3

class AskResponse(BaseModel):
    answer: str
    chunks: list[str]
    source: list[str]

embedding_model = EmbeddingModel()
index = faiss.read_index('datas/processed/index/index1.index')
with open('datas/processed/content/content1.json', 'r', encoding = 'utf-8') as f:
    documents = json.load(f)

@app.post('/ask')
async def ask(ask_request: AskRequest) -> AskResponse:
    question_embedding = np.array(embedding_model.get_embedding([ask_request.question])).astype('f4')
    faiss.normalize_L2(question_embedding)
    D, I = index.search(question_embedding, k = ask_request.top_k)
    context = []
    source = []
    for i in I[0]:
        if i == -1:
            continue
        context.append(documents[i]['content'])
        source.append(documents[i]['source'])
    answer = llm(ask_request.question, '\n'.join(context))

    return AskResponse(
        answer = answer,
        chunks = context,
        source = list(set(source))
    )