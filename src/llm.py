from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def llm(history: list[dict], model: str = 'qwen3-vl-flash'):


    completions = client.chat.completions.create(
        model = model,
        messages = history,
        temperature = 0.1
    )

    return completions.choices[0].message.content