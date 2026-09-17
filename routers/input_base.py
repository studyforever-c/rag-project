from fastapi import APIRouter, UploadFile, File
from src.temp_file import temp_file
from src.persist import persist
from src.rerank import rerank
from commons.prompts import system_prompt, user_prompt
from commons.pydantic_models import AskRequest, AskResponse
from faiss import IndexFlatIP
from src.rewrite import rewrite
# from config import Dimension # 本地模型不适用
from src.llm import llm
from rank_bm25 import BM25Okapi
from src.hybrid_search import hybrid_search
import numpy as np
import jieba

router = APIRouter()


documents = []
bm25 = None

index = None
embeddings = []
@router.post('/upload')
async def upload(files: list[UploadFile] = File(...)):
    global bm25
    global index
    for file in files:
        tmp_path = await temp_file(file)
        try:
            d, e = persist(str(tmp_path), False, False)
            documents.extend(d)
            embeddings.extend(e)
        except Exception as e:
            print('出错啦！！！', e)
        finally:
            tmp_path.unlink()
    bm25 = BM25Okapi([jieba.lcut(d['content']) for d in documents])
    if not index:
        index = IndexFlatIP(len(embeddings[0]))
    index.add(np.array(embeddings).astype('f4'))

history = [{'role': 'system', 'content': system_prompt}]
show_history = []
@router.post('/input_base')
async def input_base(ask_request: AskRequest) -> AskResponse:
    question = rewrite(ask_request.question, history)
    may_need = []  # 混合检索之后获得的documents
    if index is not None:
        I = hybrid_search(question, bm25, index)
        for i in I:
            may_need.append(documents[i])
    context = rerank(question, may_need)

# 不使用重排序
#     context = []
#     if index is not None:
#         I = hybrid_search(question, bm25, index)
#     for i in I:
#         context.append(documents[i]['content'])

    history.append({'role': 'user',
                    'content': user_prompt.format(context=context, question=question)})
    show_history.append({'role': 'user',
                         'content': ask_request.question})
    answer = llm(history)
    history.append({'role': 'assistant',
                    'content': answer})
    show_history.append({'role': 'assistant',
                         'content': answer})
    return AskResponse(
        history = show_history
    )

@router.post('/renew')
async def renew():
    global index
    global bm25
    history.clear()
    show_history.clear()
    documents.clear()
    embeddings.clear()
    index = None
    bm25 = None