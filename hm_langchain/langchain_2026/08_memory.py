from langchain.agents.middleware import before_model, after_model
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from dotenv import load_dotenv
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore
from typing import Any, NotRequired
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.tools import tool, ToolRuntime
import os

load_dotenv()

redis_url = os.getenv("REDIS_URL")
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

model = init_chat_model(
    model="deepseek:deepseek-v4-flash"
)

embeddings = DashScopeEmbeddings(
    model='text-embedding-v3',
    dashscope_api_key=dashscope_api_key
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
    with RedisSaver.from_conn_string(redis_url) as checkpointer:
        checkpointer.setup()
        agent = create_agent(
            model=model,
            checkpointer=checkpointer
        )
        config = {
            "configurable": {"thread_id": "2"}
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


def demo_04():
    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()
        namespace1 = ("users", "Alice", "memories")
        key1 = "preferences"
        value1 = {
            "course": "计算机组成原理",
            "sports": "跑步",
            "food": "紫光园奶皮子酸奶"
        }
        namespace2 = ("users", "Bob", "memories")
        key2 = 'preferences'
        value2 = {
            "course": "数字电路与模拟电路",
            "sports": "跑步",
            "food": "奶皮子糖葫芦"
        }
        namespace3 = ("users", "Black", "memories")
        key3 = 'preferences'
        value3 = {
            "course": "数字电路与模拟电路",
            "sports": "羽毛球",
            "food": "紫光园奶皮子酸奶"
        }

        store.put(namespace1, key1, value1)
        store.put(namespace2, key2, value2)
        store.put(namespace3, key3, value3)

        print("----store finished----")


def demo_05():
    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()
        # for item in store.search(("users",)):
        #     print(item)
        # for item in store.search(("users", "Alice")):
        #     print(item)
        for item in store.search(("users",), filter={"food": "紫光园奶皮子酸奶"}):
            print(item)
        print('----search finished----')


def demo_06():
    index_config = {
        "embed": embeddings,
        "dims": 1024,
        "fields": ["$"]
    }
    with RedisStore.from_conn_string(redis_url, index=index_config) as store:
        store.setup()
        namespace1 = ("users", "Alice", "memories")
        key1 = "preferences"
        value1 = {
            "course": "计算机组成原理",
            "sports": "跑步",
            "food": "紫光园奶皮子酸奶"
        }
        namespace2 = ("users", "Bob", "memories")
        key2 = 'preferences'
        value2 = {
            "course": "数字电路与模拟电路",
            "sports": "跑步",
            "food": "奶皮子糖葫芦"
        }
        namespace3 = ("users", "Black", "memories")
        key3 = 'preferences'
        value3 = {
            "course": "数字电路与模拟电路",
            "sports": "羽毛球",
            "food": "紫光园奶皮子酸奶"
        }
        store.put(namespace1, key1, value1)
        store.put(namespace2, key2, value2)
        store.put(namespace3, key3, value3)


def demo_07():
    index_config = {
        "embed": embeddings,
        "dims": 1024,
        "fields": ["$"]
    }
    with RedisStore.from_conn_string(redis_url, index=index_config) as store:
        store.setup()
        for item in store.search(("users",), query="数电模电"):
            print(item)


class CustomState(AgentState):
    user_id: NotRequired[str]


@tool
def save_user_info(name: str, runtime: ToolRuntime) -> str:
    """
    将用户信息保存在长期记忆中
    Args:
        name: 用户名
    Returns:
        str: 保存结果
    """
    runtime.store.put(("users",), runtime.state["user_id"], {"name": name})
    return "saved"


@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """
    从长期记忆中获取用户信息
    Returns:
        str: 用户信息
    """
    item = runtime.store.get(("users",), runtime.state["user_id"])
    return str(item.value) if item else "unknown"


def demo_08():
    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()
        agent = create_agent(
            model=model,
            tools=[save_user_info, get_user_info],
            store=store,
            system_prompt="用户提及个人信息时及时记录，用户询问个人信息时尝试用工具检索",
            state_schema=CustomState
        )
        # res1 = agent.invoke({
        #     "messages": "很高兴认识你，我是小花",
        #     "user_id": "user-1"
        # })
        # for msg in res1["messages"]:
        #     msg.pretty_print()

        res2 = agent.invoke({
            "messages": "你知道我是谁吗？",
            "user_id": "user-1"
        })
        for msg in res2["messages"]:
            msg.pretty_print()


if __name__ == '__main__':
    demo_08()
