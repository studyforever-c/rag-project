from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def llm(question: str, context: str) -> str:
    prompt = f"""
你是一个专业、严谨的问答助手。

你的回答必须严格基于下面提供的「参考资料」，不得凭空编造。
如果参考资料中找不到答案，请直接说“根据提供的资料无法回答该问题”，不要尝试猜测。

回答要求：
1. 优先使用参考资料中的原文信息，用自己的话组织语言，不要照抄。
2. 如果问题涉及多个资料片段，请综合归纳后给出完整回答。
3. 回答时尽量简洁清晰，避免冗余。
4. 如果参考资料中有矛盾信息，请指出矛盾并分别说明。

参考资料：
{context}

用户问题：
{question}

符合条件的回答：
"""

    completions = client.chat.completions.create(
        model = 'qwen3-vl-flash',
        messages = [
            {'role': 'user', 'content': prompt}
        ],
        temperature = 0.1
    )

    return completions.choices[0].message.content