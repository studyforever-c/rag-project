from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download
from dotenv import load_dotenv
from config import LOCAL_MODEL_ID, LOCAL_MODEL_PATH, Dimension
import dashscope
import os

load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
BASE_URL = os.getenv('OPENAI_BASE_URL')
dashscope.api_key = API_KEY
dashscope.base_url = BASE_URL


class EmbeddingModel:
    """文本转向量模型"""
    def __init__(self,
                 model_name: str = LOCAL_MODEL_ID,
                 path: str = LOCAL_MODEL_PATH,
                 is_api: bool = False):

        self.is_api = is_api
        if not is_api:
            model = snapshot_download(
                model_id=model_name,
                local_dir=path,
            )
            self.model = SentenceTransformer(model)
        else:
            self.model_name = model_name

    def get_embedding(self, text: list[str]) -> list[list[float]]:
        if not self.is_api:
            return self.model.encode(
                inputs = text,
                normalize_embeddings = True
            ).tolist()
        else:
            ret = []
            # 很神奇，qwen的这个embedding模型一次只能处理25条数据，搞半天才发现
            for i in range(0, len(text), 25):
                head = i
                tail = min(i+25, len(text))
                ret.extend([
                    item['embedding']
                    for item in dashscope.TextEmbedding.call(
                        model = self.model_name,
                        input = text[head:tail],
                        dimension = Dimension,
                    ).output['embeddings']
                ])
            return ret