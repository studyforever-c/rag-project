import os
import faiss
import json
import numpy as np

from src.models import get_local_embedding_model, get_online_embedding_model
from src.parse import extract_file_text
from src.chunk import chunking
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
    i = 0
    if is_dir:
        for root, _, files in os.walk(path):
            for file in files:
                file = os.path.join(root, file)
                content = extract_file_text(file)
                chunks = chunking(content)
                embeddings.extend(local_embedding_model.get_embedding(chunks))
                for chunk in chunks:
                    documents.append({
                        'id': i,
                        'content': chunk,
                        'source': file
                    })
                    i += 1
    else:
        content = extract_file_text(path)
        chunks = chunking(content)
        # embeddings.extend(online_embedding_model.get_embedding(chunks))
        embeddings.extend(local_embedding_model.get_embedding(chunks))

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
    persist(
        r'datas\raw',
        True,
        True,
        'index.index',
        'content.json',
    )
