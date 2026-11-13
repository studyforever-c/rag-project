from sentence_transformers import CrossEncoder
from modelscope import snapshot_download

model = snapshot_download(
    model_id = 'BAAI/bge-reranker-base',
    local_dir = r'D:\big_mantou_tools\transformers_models\models\bge-reranker-base'
)

rerank_model = CrossEncoder(model)

def rerank(question: str,
                    may_need: list[dict],
                    k: int = 15) -> list[dict]:
    pairs = [
        (question, doc['content'])
        for doc in may_need
    ]

    rerank_scores = rerank_model.predict(pairs)

    for i, doc in enumerate(may_need):
        doc['rerank_score'] = rerank_scores[i]

    may_need.sort(key = lambda x: x['rerank_score'], reverse = True)
    return may_need[: k]