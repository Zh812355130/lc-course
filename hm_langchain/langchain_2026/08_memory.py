from langchain.agents.middleware import before_model
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from dotenv import load_dotenv
from langgraph.checkpoint.redis import RedisSaver
from typing import Any

load_dotenv()

model = init_chat_model(
    model="deepseek:deepseek-v4-flash"
)


def demo_01():
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=model,
        checkpointer=checkpointer
    )
    config = {
        "configurable": {"thread_id": "1"}
    }

    print("\n 第一轮对话:")
    response = agent.invoke({
        "messages": [HumanMessage("我叫张三")]},
        config=config
    )
    print(f"Agent: {response['messages'][-1].content}")
    print("\n 第二轮对话:")
    response = agent.invoke({
        "messages": [HumanMessage("我叫什么？")]},
        config=config
    )
    print(f"Agent: {response['messages'][-1].content}")


def demo_02():
    # 39.102.72.17  zh@123
    with RedisSaver.from_conn_string("redis://:zh%40123@39.102.72.17:6379") as checkpointer:
        checkpointer.setup()
        agent = create_agent(
            model=model,
            checkpointer=checkpointer
        )
        config = {
            "configurable": {"thread_id": "1"}
        }
        response = agent.invoke({
            "messages": [HumanMessage("我叫张三")]},
            config=config
        )
        print("------------第一次调用-------------")
        for msg in response['messages']:
            msg.pretty_print()
        res = agent.invoke({
            "messages": [HumanMessage("我叫什么？")]
        }, config=config)
        print("------------第二次调用-------------")
        for msg in res['messages']:
            msg.pretty_print()


@before_model
def trim_message(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state['messages']
    if len(messages) <= 3:
        return None


if __name__ == '__main__':
    demo_02()
