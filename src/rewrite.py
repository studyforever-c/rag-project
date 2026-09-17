"""
对问题重写
"""
from commons.prompts import rewrite_system, rewrite_prompt
from src.llm import llm
def rewrite(question: str, history: list[dict]) -> str:
    # 根据最近三轮对话对问题进行改写
    history_text = '\n'.join([
        f"{h['role']}: {h['content']}"
        for h in history[-6:]
    ])

    messages = [
        {'role': 'system', 'content': rewrite_system},
        {'role': 'user', 'content': rewrite_prompt.format(
            history = history_text,
            question = question)}
    ]
    resp = llm(messages)
    return question if not resp else resp