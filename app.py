import numpy as np
import faiss
import json
from pydantic_models import AskRequest, AskResponse
from fastapi import FastAPI
from src.embed import EmbeddingModel
from src.llm import llm
app = FastAPI(title = '基于RAG的本地知识库问答助手')


embedding_model = EmbeddingModel()
index = faiss.read_index('datas/processed/index/index1.index')
with open('datas/processed/content/content1.json', 'r', encoding = 'utf-8') as f:
    documents = json.load(f)

system_prompt = f"""
你是一个专业、严谨的问答助手。
你的回答必须严格基于下面提供的「参考资料」，不得凭空编造。
如果参考资料中找不到答案，请直接说“根据提供的资料无法回答该问题”，不要尝试猜测。
回答要求：
1. 优先使用参考资料中的原文信息，用自己的话组织语言，不要照抄。
2. 如果问题涉及多个资料片段，请综合归纳后给出完整回答。
3. 回答时尽量简洁清晰，避免冗余。
4. 如果参考资料中有矛盾信息，请指出矛盾并分别说明。
5. 每条来源必须在新的一行展示。
"""

user_prompt = """
参考资料：
{context}
用户问题：
{question}
"""

"""
history是投喂给大模型的信息，show_history是展示给用户的信息，
之所以搞两个是因为自己测试时发现都用history的话会导致用户界面信息显示混乱，
后期如果对历史记录进行压缩的话也会导致用户看到的内容发生变化
"""
history = [{'role': 'system', 'content': system_prompt}]
show_history = []
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
    source = list(set(source))
    content = '内容'+'\n'.join(context)+ '\n\n来源' + '\n'.join(source)

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