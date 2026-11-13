import json
import faiss
import jieba
from src.hybrid_search import hybrid_search
from src.rerank import rerank
from rank_bm25 import BM25Okapi
from commons.prompts import system_prompt, user_prompt
from src.llm import llm
from src.rewrite import rewrite



documents_path = 'datas/processed/content/content_1.json'
index_path = 'datas/processed/index/index_1.index'

with open(documents_path, 'r', encoding='utf-8') as f:
    documents = json.load(f)
index = faiss.read_index(index_path)

bm25 = BM25Okapi(
    jieba.lcut(doc['content'])
    for doc in documents
)


def retrieve(question: str) -> list[int]:
    """
    检索函数
    """
    hybrid_result = [
        documents[i]
        for i in hybrid_search(question, bm25, index)
    ]
    rerank_result = rerank(question, hybrid_result)
    return [doc['id'] for doc in rerank_result]

def generate(context: str, question: str):
    history = [{'role': 'system', 'content': system_prompt},
               {'role': 'user', 'content': user_prompt.format(context=context, question=question)}]
    answer = llm(history)
    return answer

    history = [{'role': 'system', 'content': system_prompt},
               {'role': 'user', 'content': user_prompt.format(context=context, question=question)}]
    answer = llm(history)
def main():
    test_set_path = 'evals/test_set.json'
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)
    result = []
    for i, test in enumerate(test_set):
        question = test['query']
        question = "为这个句子生成表示以用于检索相关文章:" + rewrite(question, [])
        ids = retrieve(question)
        context = '\n'.join([f'资料{documents[int(j)]['content']}, 来源：{documents[int(j)]['source']}'
                             for j in ids])
        chunk_list = [documents[int(j)]['content'] for j in ids]

        result.append({
            'question': question,
            'ids': ids,
            'chunk_list': chunk_list,
            'answer': generate(context, question)
        })
        print(f"进度：{i+1} / {len(test_set)}")
    with open('evals/results/retrieve_result.json', 'w', encoding = 'utf-8') as f:
        json.dump(result, f, ensure_ascii = False, indent = 2)
if __name__ == '__main__':
    main()