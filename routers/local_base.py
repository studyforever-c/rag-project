import jieba
import numpy as np
import faiss
import json
from commons.pydantic_models import AskRequest, AskResponse
from fastapi import APIRouter
from src.models import get_local_embedding_model
from src.llm import llm
from commons.prompts import system_prompt, user_prompt
from faiss import IndexFlatIP
from src.hybrid_search import hybrid_search
from rank_bm25 import BM25Okapi

router = APIRouter()
local_embedding_model = get_local_embedding_model()

index = faiss.read_index('datas/processed/index/index1.index')
with open('datas/processed/content/content1.json', 'r', encoding = 'utf-8') as f:
    documents = json.load(f)

bm25 = BM25Okapi(
    jieba.lcut(doc['content'])
    for doc in documents
)

"""
history是投喂给大模型的信息，show_history是展示给用户的信息，
之所以搞两个是因为自己测试时发现都用history的话会导致用户界面信息显示混乱，
后期如果对历史记录进行压缩的话也会导致用户看到的内容发生变化
"""
history = [{'role': 'system', 'content': system_prompt}]
show_history = []
@router.post('/local_base')
async def local_base(ask_request: AskRequest) -> AskResponse:
    question_embedding = np.array(local_embedding_model.get_embedding([ask_request.question])).astype('f4')
    faiss.normalize_L2(question_embedding)
    # D, I = index.search(question_embedding, k = ask_request.top_k)
    I = hybrid_search(ask_request.question,
                      bm25,
                      index)
    context = []
    # for i in I[0]:
    #     if i == -1:
    #         continue
    #     context.append(documents[i]['content'])
    for i in I:
        context.append(documents[i]['content'])
    content = '内容'+'\n'.join(context)

    # 大模型需知道参考内容，用户只需看到问题
    history.append({'role': 'user',
                    'content': user_prompt.format(context = content, question = ask_request.question)})
    show_history.append({'role': 'user',
                    'content': ask_request.question})

    answer = llm(history)
    history.append({'role': 'assistant', 'content': answer})
    show_history.append({'role': 'assistant', 'content': answer})

    return AskResponse(
        history = show_history
    )

@router.post('/renew')
async def renew():
    global index
    history.clear()
    history.append({'role': 'system', 'content': system_prompt})
    show_history.clear()
    documents.clear()
    index = IndexFlatIP(index.d)