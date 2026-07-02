from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from rich import print as rprint

load_dotenv(override=True)


def demo_01():
    model = init_chat_model(model="deepseek:deepseek-v4-flash")
    prompt = "你好"
    response = model.invoke(prompt)
    rprint(response)


def demo_02():
    model = init_chat_model(model="deepseek:deepseek-v4-flash")
    prompt = "写一首关于夏天的古诗"
    response = model.invoke(prompt)
    rprint(response)


def demo_03():
    model = init_chat_model(model="deepseek:deepseek-v4-flash")
    prompt = "写一首关于动物的笑话"
    config = {
        "run_name": "joke_generation",
        "tags": ["joke", "animal"],
        "metadata": {
            "user_id": "huan_001",
            "session_id": "session_914123128371231"
        }
    }
    response = model.invoke(prompt, config=config)
    rprint(response)


if __name__ == '__main__':
    demo_03()
