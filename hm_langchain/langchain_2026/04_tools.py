from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from rich import print as rprint
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv(override=True)


# pydantic 风格参数
class WeatherInput(BaseModel):
    city: str = Field(
        default="北京",
        description="城市"
    )
    include_forecast: bool = Field(
        default=False,
        description="是否包含明日天气预报"
    )


# name_or_callable 可以更改工具名称
# @tool("getWeather")
@tool("get_weather_and_forecast",
      description="根据城市名称查询天气信息,可以包含明日天气预报",
      args_schema=WeatherInput)
def get_weather(city: str, include_forecast: bool) -> str:
    """
    获取指定城市的天气信息
    参数：
        city: 城市名称，如"北京","上海"
    返回：
        天气信息字符串
    """
    res = city + "晴天，温度15℃"
    if include_forecast:
        res += "，明天天气晴转多云，最低气温13℃，最高气温17℃"
    return res


def demo_01():
    result = get_weather.invoke({"city": "北京"})
    print(result)


def demo_02():
    model = init_chat_model(
        model='deepseek:deepseek-v4-flash',
        extra_body={"thinking": {"type": "disabled"}}
    )
    # model_with_tool = model.bind_tools([get_weather], tool_choice="get_weather_and_forecast")
    model_with_tool = model.bind_tools([get_weather], tool_choice="auto")
    messages = [
        HumanMessage(content="告诉我北京今天天气如何？明天呢？")
    ]
    response = model_with_tool.invoke(messages)
    messages.append(response)
    tool_calls = response.tool_calls
    for tool_call in tool_calls:
        if tool_call["name"] == "get_weather_and_forecast":
            tool_msg = get_weather.invoke(tool_call)
            messages.append(tool_msg)
    final_response = model_with_tool.invoke(messages)
    messages.append(final_response)
    for msg in messages:
        msg.pretty_print()


if __name__ == '__main__':
    demo_02()
