from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download
from dotenv import load_dotenv
from openai import OpenAI
from config import LOCAL_MODEL_ID, LOCAL_MODEL_PATH

load_dotenv()

class EmbeddingModel:
    """
    文本转向量类，默认使用本地模型进行转换
    """
    def __init__(self, model_name: str = LOCAL_MODEL_ID, path: str = LOCAL_MODEL_PATH, is_api: bool = False):
        self.path = path #,（震惊，这里加个“,”就会把变量转为tuple类型）
        self.is_api = is_api
        if self.is_api:
            self.client = OpenAI()
            self.model_name = model_name
        else:
            model = snapshot_download(
                model_id=model_name,
                local_dir=self.path,
            )
            self.model = SentenceTransformer(model)

    def get_embedding(self, text: list[str]) -> list[list[float]]:
        if self.is_api:
            completions = self.client.embeddings.create(
                model = self.model_name,
                input = text
            )
            return [data.embedding for data in completions.data]
        else:
            return self.model.encode(
                inputs = text,
                normalize_embeddings = True
            ).tolist()