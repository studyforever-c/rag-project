"""
混合检索实现
"""
from rank_bm25 import BM25Okapi
from faiss import IndexFlatIP
from src.models import get_local_embedding_model
import numpy as np
import jieba
import faiss

embedding_model = get_local_embedding_model()
def hybrid_search(question: str,
                  bm25: BM25Okapi|None,
                  index: IndexFlatIP,
                  k: int = 30,
                  callback: int = 50) -> list[int]:
    question_embedding = np.array(embedding_model.get_embedding([question])).astype('f4')
    
    # 稠密检索
    D, I = index.search(question_embedding, callback)
    final_scores = {}
    for i in I[0]:
        if i == -1:
            continue
        final_scores[i] = final_scores.get(i, 0) + 1 / (60 + i)

    # 稀疏检索
    if bm25 is not None:
        scores = bm25.get_scores(jieba.lcut(question))
        for i in np.argsort(scores, descending = True)[:callback]:
            final_scores[i] = final_scores.get(i, 0) + 1 / (60 + i)

    indices = sorted(final_scores.items(),
                     key = lambda x: x[1],
                     reverse = True)
    return [item[0] for item in indices[:k]]