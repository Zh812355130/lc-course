from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy, AutoStrategy
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from rich import print as rprint
from pydantic import BaseModel, Field
from typing import Union

load_dotenv()


class ConcatInfo(BaseModel):
    """用户的联系方式"""
    name: str = Field(description="用户姓名")
    email: str = Field(description="用户邮箱")
    phone: str = Field(description="用户电话")


class EventInfo(BaseModel):
    """事件信息"""
    event_name: str = Field(description="事件名称")
    event_content: str = Field(description="事件简介")
    date: str = Field(description="事件时间")


@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """
    天气查询工具

    Args:
        city (str): 城市名称
    """
    return f'{city}的天气晴朗，25℃'


@tool(parse_docstring=True)
def get_news():
    """
    新闻查询工具

    """
    return "近期，受全球储蓄芯片短缺等多重因素影响，多地回收商称废旧手机回收市场迎来“火热潮”，回收价格普遍上涨，旧手机成“香饽饽”。"


def demo_01():
    agent = create_agent("deepseek-v4-flash")
    print(type(agent))


def demo_02():
    model = init_chat_model("deepseek:deepseek-v4-flash")
    agent = create_agent(model)
    print(type(agent))
    response = agent.invoke({"messages": ["你好"]})
    print(type(response))
    rprint(response)


def demo_03():
    agent = create_agent(
        model="deepseek:deepseek-v4-flash",
        tools=[get_weather, get_news])
    res = agent.invoke({
        "messages": [
            {"role": "user", "content": "上海的天气如何？今天有哪些新闻？"}
        ]
    })
    rprint(res)


def demo_04():
    agent = create_agent(
        model="deepseek:deepseek-v4-flash",
        name="chat_assistant",
        system_prompt="回答简洁明了，尽量使用中文回答"
    )
    res = agent.invoke({"messages": ['hello']})
    for msg in res["messages"]:
        msg.pretty_print()


def demo_05():
    model = init_chat_model(
        model="deepseek:deepseek-v4-flash",
        extra_body={"thinking": {"type": "disabled"}}
    )
    agent = create_agent(
        model=model,
        # response_format=ProviderStrategy(ConcatInfo)
        # response_format=ToolStrategy(ConcatInfo)
        # response_format=AutoStrategy(ConcatInfo)
        # response_format=ConcatInfo
        response_format=ToolStrategy(Union[ConcatInfo, EventInfo], tool_message_content="成功获取到结构化信息")
    )
    res = agent.invoke({"messages": [
        HumanMessage("从这段话中抽取结构化信息：小明的邮箱地址为：shkstart@atguigu.com，手机号：12345678912")
    ]})
    for msg in res['messages']:
        msg.pretty_print()
    print(res["structured_response"])
    print("-" * 20)
    res = agent.invoke({"messages": [
        HumanMessage("从这段话中抽取结构化信息：2026年高考报名人数突破1200万")
    ]})
    for msg in res['messages']:
        msg.pretty_print()
    print(res["structured_response"])


def demo_06():
    agent = create_agent(
        model="deepseek:deepseek-v4-flash",
        tools=[get_weather, get_news]
    )
    for chunk in agent.stream(
            {"messages": [HumanMessage("上海的天气如何？今天有哪些新闻？")]},
            # stream_mode="values"
            # stream_mode="updates"
            stream_mode="messages"
    ):
        rprint(chunk)
        print("-" * 50)


if __name__ == '__main__':
    demo_06()
