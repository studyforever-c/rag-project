from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download
from dotenv import load_dotenv
from config import LOCAL_MODEL_ID, LOCAL_MODEL_PATH

load_dotenv()

class EmbeddingModel:
    """
    文本转向量，使用本地模型进行转换
    """
    def __init__(self, model_name: str = LOCAL_MODEL_ID, path: str = LOCAL_MODEL_PATH):
        model = snapshot_download(
            model_id=model_name,
            local_dir=path,
        )
        self.model = SentenceTransformer(model)

    def get_embedding(self, text: list[str]) -> list[list[float]]:
        return self.model.encode(
            inputs = text,
            normalize_embeddings = True
        ).tolist()