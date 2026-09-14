from fastapi import APIRouter, UploadFile, File
from src.temp_file import temp_file
from src.persist import persist
from commons.prompts import system_prompt, user_prompt
from commons.pydantic_models import AskRequest, AskResponse
from faiss import IndexFlatIP
from config import Dimension
from src.llm import llm
from src.models import get_online_embedding_model
import faiss
import numpy as np

online_embedding_model = get_online_embedding_model()

router = APIRouter()


documents = []
index = IndexFlatIP(Dimension)
embeddings = []
@router.post('/upload')
async def upload(files: list[UploadFile] = File(...)):
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
    index.add(np.array(embeddings).astype('f4'))

history = [{'role': 'system', 'content': system_prompt}]
show_history = []
@router.post('/input_base')
async def input_base(ask_request: AskRequest) -> AskResponse:
    question_embedding = np.array(online_embedding_model.get_embedding([ask_request.question])).astype('f4')
    faiss.normalize_L2(question_embedding)
    D, I = index.search(question_embedding, k = ask_request.top_k)
    context = []
    for i in I[0]:
        if i == -1:
            continue
        context.append(documents[i]['content'])
    history.append({'role': 'user',
                    'content': user_prompt.format(context=context, question=ask_request.question)})
    show_history.append({'role': 'user',
                         'content': ask_request.question})
    answer = llm(history)
    history.append({'role': 'user',
                    'content': answer})
    show_history.append({'role': 'user',
                         'content': answer})
    return AskResponse(
        history = show_history
    )

@router.post('/renew')
async def renew():
    global index
    history.clear()
    show_history.clear()
    documents.clear()
    embeddings.clear()
    index = IndexFlatIP(Dimension)

