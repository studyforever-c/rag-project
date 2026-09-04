import os
import faiss
import json
import numpy as np

from src.embed import EmbeddingModel
from src.parse import extract_file_text
from src.chunk import chunking
embedding_model = EmbeddingModel()

def persist(path: str, index_file_name: str, content_file_name: str) -> None:

    # 获取文本
    documents = []
    embeddings = []
    i = 0
    for root, _, files in os.walk(path):
        for file in files:
            path = os.path.join(root, file)
            content = extract_file_text(path)
            chunks = chunking(content)
            embeddings.extend(embedding_model.get_embedding(chunks))
            for chunk in chunks:
                documents.append({
                    'id': i,
                    'content': chunk,
                    'source': path
                })
                i += 1
    if not documents:
        print('所选目录/文件为空')
        return

    # 构建索引
    dimension = len(embeddings[0])
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings).astype('f4'))

    # 确保文件存储位置存在
    os.makedirs('datas/processed/index', exist_ok = True)
    os.makedirs('datas/processed/content/', exist_ok = True)

    index_path = os.path.join('datas/processed/index', index_file_name)
    content_path = os.path.join('datas/processed/content', content_file_name)


    faiss.write_index(index, index_path)
    with open(content_path, 'w', encoding = 'utf-8') as f:
        json.dump(documents, f, ensure_ascii = False, indent = 2)


    print(f"""
已完成faiss数据库构建：
索引存储位置：{index_path}，
文本切片存储位置：{content_path}
        """
    )

if __name__ == '__main__':
    persist(
        'datas/raw',
        'index1.index',
        'content1.json'
    )