from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import base64
from template import PromptLibrary

load_dotenv(override=True)

ali_key = os.getenv("DASHSCOPE_API_KEY")
ali_url = os.getenv("DASHSCOPE_BASE_URL")


def keep_recent_messages(messages, max_pairs=3):
    system_msgs = [m for m in messages if m.get('role') == 'system']
    conversation_msgs = [m for m in messages if m.get('role') != 'system']
    recent_msgs = conversation_msgs[-(max_pairs * 2):]
    return system_msgs + recent_msgs


def encode_image(img_path, img_type='jpeg'):
    with open(img_path, 'rb') as img_file:
        return f"data:image/{img_type};base64,{base64.b64encode(img_file.read()).decode('utf-8')}"


def demo_01():
    model = init_chat_model('deepseek:deepseek-v4-flash')
    msgs = [{'role': 'system', 'content': '你是python导师'}, {'role': 'user', 'content': '什么是列表？用一句话解释'}]
    r1 = model.invoke(msgs)
    msgs.append({'role': 'assistant', 'content': r1.content})

    msgs.append({'role': 'user', 'content': '列表和元祖的区别？用一句话解释'})
    r2 = model.invoke(msgs)
    msgs.append({'role': 'assistant', 'content': r2.content})

    msgs.append({'role': 'user', 'content': '什么是字典？用一句话解释'})
    r3 = model.invoke(msgs)
    msgs.append({'role': 'assistant', 'content': r3.content})

    print(f'原始消息数：{len(msgs)}')
    print(msgs)
    optimized = keep_recent_messages(msgs, max_pairs=2)
    print(f'优化后消息数：{len(optimized)}')
    print(optimized)

    optimized.append({'role': 'user', 'content': '我的第一个问题是什么？'})
    response = model.invoke(optimized)
    print(f'\nAI回复：{response.content}')


def demo_02():
    model = ChatOpenAI(
        api_key=ali_key,
        base_url=ali_url,
        model='qwen3.6-plus'
    )

    img_path = 'image_test.png'
    base64_image = encode_image(img_path)

    response = model.invoke([
        HumanMessage(
            content=[
                {'type': 'text', 'text': '这张图里有什么？'},
                {'type': 'image_url', 'image_url': {'url': base64_image}}
            ]
        )
        #     HumanMessage(
        #         content_blocks=[
        #             {
        #                 'type': 'text',
        #                 'text': '这张图里有什么？'
        #             },
        #             {
        #                 'type': 'image',
        #                 'base64': base64_image,
        #                 'mime_type': 'image/png'
        #             }
        #         ]
        #     )
    ])
    print(response.content)


def demo_03():
    model = init_chat_model(
        model='deepseek:deepseek-v4-flash',
        extra_body={"thinking": {"type": "enabled"}}
    )
    response = model.invoke("你好，一句话回答")
    print('=' * 20, 'response', '=' * 20)
    print(response)
    print('=' * 20, 'response.content', '=' * 20)
    print(response.content)
    print('=' * 20, 'response.content_blocks', '=' * 20)
    print(response.content_blocks)


def demo_04():
    # chat_template = ChatPromptTemplate.from_messages([
    #     ("system", "你是一个AI机器人，你的名字是{name}。"),
    #     ("human", "你好，最近怎么样？"),
    #     ("ai", "我很好，谢谢"),
    #     ("human", "{user_input}")
    # ])

    chat_template = ChatPromptTemplate([
        ("system", "你是一个AI机器人，你的名字是{name}。"),
        ("human", "你好，最近怎么样？"),
        ("ai", "我很好，谢谢"),
        ("human", "{user_input}")
    ])

    # prompt = chat_template.invoke({"name": "小张", "user_input": "你叫什么名字？"})
    # prompt = chat_template.format(name="小张", user_input= "你叫什么名字？")
    prompt = chat_template.format_messages(name="小张", user_input="你叫什么名字？")
    print(type(prompt))
    print(prompt)
    # print(len(prompt.to_messages()))


def demo_05():
    model = init_chat_model('deepseek:deepseek-v4-flash')
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个数学家，你可以计算任何算式"),
        ("human", "{text}")
    ])
    prompt = chat_prompt.invoke({
        "text": "我今年18岁，我的舅舅今年38岁，我的爷爷今年72岁，我和舅舅一共多少岁了？"
    })
    response = model.invoke(prompt)
    print(response.content)


def demo_06():
    template = ChatPromptTemplate.from_messages([
        ("system", "你是{role},目标用户是{audience}"),
        ("user", "{task}")
    ])

    customer_template = template.partial(role="客服专员", audience="普通用户")

    messages = customer_template.invoke({"task", "解释退款政策"})
    print(messages)


def demo_07():
    # template = ChatPromptTemplate.from_messages([
    #     ("system", "你是一个有用的AI助手"),
    #     ("placeholder", "{conversation}")
    # ])
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的AI助手"),
        MessagesPlaceholder(variable_name="conversation"),
        ("human", "{question}")
    ])
    prompt = template.invoke({
        "conversation": [
            ("human", "你好"),
            ("ai", "今天我能帮你做什么？"),
            ("human", "你能给我一个冰淇淋吗？"),
            ("ai", "抱歉，我没有这样的能力。"),
        ],
        "question": "今天天气怎么样"
    })
    print(prompt)


def demo_08():
    messages = PromptLibrary.TRANSLATOR.format_messages(
        source_lang="英语",
        target_lang="中文",
        text="I love programming"
    )
    print(messages)


if __name__ == '__main__':
    demo_08()
