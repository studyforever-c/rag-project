# RAG Project

## 功能介绍
- 读取文件，切分
- 向量化
- faiss检索
- llm生成回答
- streamlit页面多轮对话
- 上传文件作为知识库

## 项目结构
```
rag_project
├─commons                   # 存放一些需要重复使用的变量
    ├─prompts.py            # 提示词
    └─pydantic_model        # pydantic模型
├─datas                     # 存放数据
    ├─processed             # 存放已处理好的数据和索引
    └─raw                   # 存放原始数据
├─main                      # 前后端启动文件
    ├─__init__.py           # 声明，否则启动指令识别不了
    ├─app.py                # fastapi启动
    └─ui.py                 # streamlit启动
├─pages                     # 页面
    ├─input_base.py         # 基于用户上传文件
    └─local_base.py         # 基于本地知识库
├─routers                   # 路由，与页面文件一一对应
    ├─input_base.py         
    └─local_base.py         
├─src                       # 核心功能代码
    ├─chunk.py              # 文本切分函数
    ├─embed.py              # 嵌入模型的初始化类
    ├─hybrid_search.py      # 混合检索实现
    ├─llm.py                # 调用生成模型获得回答
    ├─models.py             # 管理模型对象
    ├─parse.py              # 文档解析函数
    ├─persist.py            # faiss持久化
    └─temp_file.py          # 构造临时文件
    
├─.env                      # 存放调用生成模型的api_key和base_url
├─config.py                 # 配置文件
├─.gitignore
└─requirements.txt           # 项目运行所需依赖
```

## 快速开始

安装依赖
```
pip install -r requirement.txt
```

修改config.py文件

使用本地知识库需要将要处理的文件存放在rag_project/datas/raw文件夹

创建.env文件并写入OPENAI_API_KEY，OPENAI_BASE_URL

构建向量数据库（仅本地知识库需要）
```
python -m src.persist
```

启动fastapi
```
fastapi dev main/app.py
```

启动streamlit，运行指令后自动打开页面
```
streamlit run main/ui.py
```

