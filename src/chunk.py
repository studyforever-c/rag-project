import re
import os
from pathlib import Path
from transformers import AutoTokenizer
from modelscope import snapshot_download
from config import LOCAL_MODEL_ID, LOCAL_MODEL_PATH

model = None
# 定义分词器，用于确定一段文本会被切分的token数
tokenizer = None
def chunking(text: str|list[str], max_token: int = 600, overlap: int = 150) -> list[str]:
    """切分函数"""
    if type(text) == list:
        text = '\n'.join(text)
    global model
    global tokenizer
    if not model:
        model = snapshot_download(
        model_id=LOCAL_MODEL_ID,
        local_dir=LOCAL_MODEL_PATH
        )
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model)

    cleaned = re.sub(r'\s+', ' ', text)
    sentences = [sentence.strip()
                 for sentence in re.split(r'(?<=[.?!。？！])', cleaned)
                 if sentence.strip()]
    chunks = []
    chunk = []
    length = 0
    for sentence in sentences:
        sentence_token = tokenizer.encode(
            sentence, add_special_tokens=False)
        sl = len(sentence_token)
        if sl <= max_token:
            if sl + length <= max_token:
                chunk.extend(sentence_token)
                length += sl
            else:
                chunks.append(chunk)
                chunk = chunk[-overlap:] + sentence_token
                length = len(chunk)
        else:
            if chunk:
                chunks.append(chunk)
            step = max_token - overlap
            for i in range(0, sl, step):
                chunk = chunks[-1][-overlap:] + sentence_token[i: i + step]
                if len(chunk) == max_token:
                    chunks.append(chunk)
                    chunk.clear()
            if not chunk:
                chunk = chunks[-1][-overlap:]
    if chunk:
        chunks.append(chunk)

    return [tokenizer.decode(chunk, skip_special_tokens=True)
            for chunk in chunks]

def get_doc(id: int,
            piece: list[str],
            filename: str,
            bian: str,
            fenbian: str,
            zhang: str,
            jie: str,
            fuze: str) -> dict:

    doc = {
        'id': str(id+1),
        'content': ''.join(piece),
        'source': ' '.join([filename, bian, fenbian, zhang, jie])
                    if not fuze else ' '.join([filename, fuze])
    }
    return doc

def law_chunk(paragraphs: list[str], path: str) -> list[dict]:
    filename = Path(path).name
    patterns = {
        'bian': re.compile(r'^第[零一二三四五六七八九十百千]+编'),
        'fenbian': re.compile(r'^第[零一二三四五六七八九十百千]+分编'),
        'zhang': re.compile(r'^第[零一二三四五六七八九十百千]+章'),
        'jie': re.compile(r'^第[零一二三四五六七八九十百千]+节'),
        'tiao': re.compile(r'^第[零一二三四五六七八九十百千]+条'),
        'mulu': re.compile(r'^目\s*录$'),
        'fuze': re.compile(r'^附\s*则$')
    }

    documents = []

    piece = []
    c_bian = ''
    c_zhang = ''
    c_jie = ''
    c_fenbian = '',
    fuze = ''
    id = 0
    for i, para in enumerate(paragraphs):

        if patterns['mulu'].match(para):
            continue

        if patterns['bian'].match(para):
            c_bian = para
            if piece:
                documents.append(get_doc(id, piece, filename, c_bian, c_fenbian, c_zhang, c_jie, fuze))
                piece.clear()
                id += 1
            c_fenbian = ''
            c_zhang = ''
            c_jie = ''
            fuze = ''
        elif patterns['fenbian'].match(para):
            c_fenbian = para
            if piece:
                documents.append(get_doc(id, piece, filename, c_bian, c_fenbian, c_zhang, c_jie, fuze))
                piece.clear()
                id += 1
            c_zhang = ''
            c_jie = ''
            fuze = ''
        elif patterns['zhang'].match(para):
            c_zhang = para
            if piece:
                documents.append(get_doc(id, piece, filename, c_bian, c_fenbian, c_zhang, c_jie, fuze))
                piece.clear()
                id += 1
            c_jie = ''
            fuze = ''
        elif patterns['jie'].match(para):
            c_jie = ''
            if piece:
                documents.append(get_doc(id, piece, filename, c_bian, c_fenbian, c_zhang, c_jie, fuze))
                piece.clear()
                id += 1
        else:
            if patterns['fuze'].match(para):
                fuze = para.replace(r'\s*', '')
            if patterns['tiao'].match(para):
                if piece:
                    documents.append(get_doc(id, piece, filename, c_bian, c_fenbian, c_zhang, c_jie, fuze))
                    id += 1
                    piece.clear()
                piece.append(para)
            else:
                if piece:
                    piece.append(para)
    if piece:
        documents.append(get_doc(id, piece, filename, c_bian, c_fenbian, c_zhang, c_jie, fuze))
    return documents