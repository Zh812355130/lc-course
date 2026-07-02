import os
from typing import Literal
from dotenv import load_dotenv
from tavily import TavilyClient
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal['general', 'news', 'finance'] = 'general',
        include_raw_content: bool = False
):
    """ Run a web search"""
    return tavily_client.search(query, max_results=max_results, include_raw_content=include_raw_content, topic=topic)


research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then 
write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, 
and whether raw content should be included.
"""

model = init_chat_model(model="deepseek-chat")

# agent = create_deep_agent(model="deepseek-chat", tools=[internet_search], system_prompt=research_instructions)
agent = create_deep_agent(model=model, tools=[internet_search], system_prompt=research_instructions)

result = agent.invoke({"messages": [
    {"role": "user", "content": "特朗普是谁？"}
]})
print(result)
print("-" * 20)
print(result["messages"][-1].content)
print("--" * 20)
for m in result['messages']:
    m.pretty_print()
