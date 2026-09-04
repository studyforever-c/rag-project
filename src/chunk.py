import re
from transformers import AutoTokenizer
from modelscope import snapshot_download
from config import LOCAL_MODEL_ID, LOCAL_MODEL_PATH


def chunking(text: str, max_token: int = 600, overlap: int = 150) -> list[str]:
    # 切分函数
    model = snapshot_download(
        model_id=LOCAL_MODEL_ID,
        local_dir=LOCAL_MODEL_PATH
    )
    # 定义分词器，用于确定一段文本会被切分的token数
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