import numpy as np
from src.models import get_online_embedding_model

model = get_online_embedding_model()

content = '你好'

resp = model.get_embedding(content)
res = np.array(resp[0])
print(np.sum(res[(res > 1) | (res < -1)]))