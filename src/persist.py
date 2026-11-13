import os
import faiss
import json
import numpy as np
import re

from src.models import get_local_embedding_model, get_online_embedding_model
from src.parse import extract_file_text
from src.chunk import chunking, law_chunk
from pathlib import Path
from config import Dimension

local_embedding_model = get_local_embedding_model()
# online_embedding_model = get_online_embedding_model()

def persist(path: str,
            is_dir: bool,
            store: bool,
            index_file_name: str = '',
            content_file_name: str = ''
            ):

    # 获取文本
    documents = []
    embeddings = []
    if is_dir:
        for root, _, files in os.walk(path):
            for file in files:
                file = os.path.join(root, file)
                content = extract_file_text(file)
                os.rename(file, os.path.join('datas/processed/origin', Path(file).name))
                documents.extend(law_chunk(content, file))
                chunks = [doc['content'] for doc in documents]
                embeddings.extend(local_embedding_model.get_embedding(chunks))
    else:
        content = extract_file_text(path)
        chunks = chunking(content)
        # embeddings.extend(online_embedding_model.get_embedding(chunks))
        embeddings.extend(local_embedding_model.get_embedding(chunks))
        i = 0
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

    if store:
        # 构建索引
        index = faiss.IndexFlatIP(len(embeddings[0]))
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
    return documents, embeddings

# 构建本地知识库使用
if __name__ == '__main__':
    path = r'datas\raw'
    label = 0
    for file in os.walk(os.path.join(path, 'index')):
        label = max(label, re.search(r'\d+', str(file)).group())
    label += 1
    persist(
        r'datas\raw',
        True,
        True,
        f'index_{label}.index',
        f'content_{label}.json',
    )
