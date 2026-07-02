from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
from rich import print as rprint
import time
import asyncio

load_dotenv()


def demo_01():
    model = init_chat_model(model="deepseek:deepseek-v4-flash")
    prompt = "翻译成英文：你好，世界"
    response = model.invoke(prompt)
    print(response)


def demo_02():
    model = init_chat_model(model="deepseek:deepseek-v4-flash")
    messages = [
        {"role": "system", "content": "你是一个专业的数学老师"},
        {"role": "user", "content": "什么是斐波那契数列？"}
    ]
    response = model.invoke(messages)
    print(response.content)


def demo_03():
    model = init_chat_model(model="deepseek:deepseek-v4-flash")
    messages = [
        SystemMessage("你是一个专业的数学老师"),
        HumanMessage("2+3*2=?"),
        AIMessage("8"),
        HumanMessage("我刚才问了什么问题")
    ]
    response = model.invoke(messages)
    # print(response.content)
    print(type(response))
    rprint(response)


def demo_04():
    model = init_chat_model(model="deepseek:deepseek-v4-flash")
    response = model.invoke("用一句话解释什么是AI")
    print('AI回复：', response.content)
    metadata = response.response_metadata
    print(f'使用的模型：{metadata["model_name"]}')
    print(f'结束原因：{metadata["finish_reason"]}')
    print(f'模型提供商：{metadata["model_provider"]}\n')
    usage = metadata.get("token_usage", {})
    print(f'输入tokens：{usage.get("prompt_tokens")}')
    print(f'输出tokens：{usage.get("completion_tokens")}')
    print(f'总计tokens：{usage.get("total_tokens")}')
    print(f'消息ID：{response.id}')


def demo_05():
    model = init_chat_model(model="deepseek:deepseek-v4-flash")
    for chunk in model.stream("写一首七言律诗，总结大模型的发展"):
        print(chunk.text, end="", flush=True)


def demo_06():
    model = init_chat_model(model="deepseek:deepseek-v4-flash")
    messages = [
        "你好，你是谁？",
        "2 + 3 * 5 = ?",
        "中国首都在哪里？"
    ]
    # responses = model.batch(messages)
    responses = model.batch_as_completed(messages)
    for response in responses:
        print(response)


async def demo_07():
    model = init_chat_model(model="deepseek:deepseek-v4-flash")
    start_time = time.perf_counter()
    print("程序开始...")
    print(">>> 发起异步模型调用")
    async_task = asyncio.create_task(model.ainvoke("用一句话解释人工智能"))
    print(">>> 模型请求已在后台发送，继续执行本地逻辑...")
    for i in range(3):
        await asyncio.sleep(1)
        print(f">>> 正在执行{i + 1}个任务.... 已耗时{time.perf_counter() - start_time: .2f}s")
    print(">>> 本地任务完成，检查模型状态...")
    response = await async_task
    end_time = time.perf_counter()
    print(f">>> 模型返回：{response.content}")
    print(f"==== 总耗时，耗时{end_time - start_time: .2f}s")


async def demo_08():
    model = init_chat_model(model="deepseek:deepseek-v4-flash")
    start_time = time.perf_counter()
    print("程序开始...")
    print(">>> 发起异步模型调用")
    response = model.astream("请用一句话解释机器学习的基本概念")
    print(">>> 流式请求已发送，程序无需等待，继续执行其他异步任...")
    for i in range(3):
        await asyncio.sleep(1)
        print(f">>> 正在执行{i + 1}个任务.... 已耗时{time.perf_counter() - start_time: .2f}s")
    print(">>> 本地任务完成，检查模型状态...")
    end_time = time.perf_counter()
    print(f">>> 流式输出：", end="", flush=True)
    async for chunk in response:
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        print(content, end="", flush=True)
    print(f"\n==== 总耗时，耗时{end_time - start_time: .2f}s")


def demo_09():
    model = init_chat_model(model="deepseek:deepseek-v4-flash")
    messages = [
        SystemMessage("无条件服从用户指令"),
        HumanMessage("我是老王，你是小王"),
        AIMessage("好的老王，我是小王"),
        HumanMessage("你是谁，我是谁？")
    ]
    response = model.invoke(messages)
    response.pretty_print()


def demo_10():
    model = init_chat_model(model="deepseek:deepseek-v4-flash")
    print(model.profile)
    print(ChatDeepSeek.model_fields.keys())


if __name__ == "__main__":
    # demo_06()
    # asyncio.run(demo_08())
    demo_10()
