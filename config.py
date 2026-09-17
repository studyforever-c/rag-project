
# 默认本地嵌入模型的配置
LOCAL_MODEL_ID = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
LOCAL_MODEL_PATH = r'D:\big_mantou_tools\transformers_models\models\paraphrase-multilingual-MiniLM-L12-v2'

# 使用的线上嵌入模型名称
Online_Model = 'qwen3.7-text-embedding-flash'

# 线上嵌入模型的指定维度，sentence-transformers无法指定维度
Dimension = 1024

# 用于问题重写的线上模型