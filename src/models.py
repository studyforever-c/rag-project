from src.embed import EmbeddingModel
from config import Online_Model

"""
管理各种模型，防止到处定义，到处重新加载模型
之前放在config文件，但是报了循环导入的错误
"""

local_embedding_model = None
online_embedding_model = None

def get_local_embedding_model():
    global local_embedding_model

    if not local_embedding_model:
        local_embedding_model = EmbeddingModel()
    return local_embedding_model

def get_online_embedding_model():
    global online_embedding_model

    if not online_embedding_model:
        online_embedding_model = EmbeddingModel(Online_Model, is_api = True)
    return online_embedding_model
