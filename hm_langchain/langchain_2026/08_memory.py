from langchain.agents.middleware import before_model, after_model
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
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
    with RedisSaver.from_conn_string("xxx") as checkpointer:
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
    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages
    return {
        "messages": [
            RemoveMessage(REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state['messages']
    print(f'-------------------{len(messages)}-------------')
    if len(messages) > 5:
        to_delete = len(messages) - 5
        return {
            "messages": [
                RemoveMessage(id=m.id) for m in messages[:to_delete]
            ]
        }
    return None


def demo_03():
    config = {"configurable": {"thread_id": "1"}}
    agent = create_agent(
        model=model,
        checkpointer=InMemorySaver(),
        middleware=[delete_old_messages]
    )
    agent.invoke({"messages": "你好，我是老王"}, config)
    agent.invoke({"messages": "从现在起，你叫小王"}, config)
    agent.invoke({"messages": "今天天气不错"}, config)
    response = agent.invoke({"messages": "告诉我，你是谁？我是谁？"}, config)
    # for msg in response["messages"]:
    #     msg.pretty_print()

    state = agent.get_state(config)
    for msg in state.values.get("messages", []):
        msg.pretty_print()


if __name__ == '__main__':
    demo_03()
