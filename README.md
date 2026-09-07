# RAG Project

## 功能介绍
- 读取文件，切分
- 向量化
- faiss检索
- llm生成回答
- streamlit页面多轮对话，刷新页面即可开启新对话

## 项目结构
```
rag_project
├─datas                     # 存放数据
    ├─processed             # 存放已处理好的数据和索引
    └─raw                   # 存放原始数据
├─src                       # 核心功能代码
    ├─chunk.py              # 文本切分函数
    ├─embed.py              # 嵌入模型的初始化类
    ├─llm.py                # 调用生成模型获得回答
    └─parse.py              # 文档解析函数
├─.env                      # 存放调用生成模型的api_key和base_url
├─app.py                    # fastapi的启动文件
├─config.py                 # 配置文件
├─persist.py                # faiss持久化
├─ui.py                     # streamlit启动文件
├─pydantic_models.py        # 定义的pydantic类，方便之后在其他文件复用
├─.gitignore
└─requirements.txt           # 项目运行所需依赖
```

## 快速开始

安装依赖
```
pip install -r requirement.txt
```

修改config.py文件

将要处理的文件存放在rag_project/datas/raw文件夹

创建.env文件并写入OPENAI_API_KEY，OPENAI_BASE_URL

构建向量数据库
```
python persist.py
```

启动fastapi
```
fastapi dev
```

启动streamlit，运行指令后自动打开页面
```
streamlit run ui.py
```

