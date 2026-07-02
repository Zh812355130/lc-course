from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware, PIIMiddleware, \
    TodoListMiddleware, before_model, after_model, before_agent, after_agent, AgentState, AgentMiddleware, hook_config, \
    wrap_model_call, wrap_tool_call, ModelRequest, ModelResponse, ExtendedModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain.tools import tool
from langchain.tools.tool_node import ToolCallRequest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langgraph.runtime import Runtime
from dotenv import load_dotenv
from rich import print as rprint
from pathlib import Path
import subprocess
from typing import Any, Callable
from datetime import datetime

load_dotenv()

custom_profile = {
    "max_input_tokens": 128_000
}

model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    profile=custom_profile,
    extra_body={"thinking": {"type": "disabled"}}
)


@tool
def get_weather(city: str, is_forcast: bool = False) -> str:
    """
    查询指定城市天气
    Args：
        city：城市名称
        is_forcast：是否查询明天天气
    """
    res = f'{city}今天天气不错'
    if is_forcast:
        res += "\n明天下雨"
    return res


@tool
def get_news() -> str:
    """
    查询当日新闻
    """
    return "美加墨世界杯今日开幕"


@tool
def read_email_tool(email_id: str) -> str:
    """
    通过邮件ID读取邮件内容
    """
    return f"邮件ID：{email_id}\n是空的"


@tool
def send_email_tool(recipient: str, subject: str, content: str) -> str:
    """
    发送邮件
    """
    print(">>> 执行发送邮件工具")
    return f'发送给{recipient}的邮件已发送，主题：{subject}，内容：{content}'


def demo_01():
    messages = [
        SystemMessage("你是个非常友好的AI助手"),
        HumanMessage("你好啊，我是老王，你是谁？"),
        AIMessage("你好老王，我是小王"),
        HumanMessage("好的小王，很高兴认识你"),
        AIMessage("你高兴得太早了"),
        HumanMessage("呵呵，你什么意思")
    ]
    agent = create_agent(
        model=model,
        middleware=[
            SummarizationMiddleware(
                model=model,
                trigger=[
                    ("tokens", 100),
                    ("messages", 6),
                    ("fraction", 0.001)
                ],
                keep=("messages", 2)
            )
        ]
    )
    res = agent.invoke({"messages": messages})
    for msg in res['messages']:
        msg.pretty_print()


def demo_02():
    agent = create_agent(
        model=model,
        tools=[get_weather, get_news, read_email_tool, send_email_tool],
        checkpointer=InMemorySaver(),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "get_weather": True,
                    "get_news": True,
                    "read_email_tool": False,
                    "send_email_tool": {
                        "allowed_decisions": ["approve", "reject"],
                        "description": "发送邮件中断了"
                    }
                },
                description_prefix="中断了~~"
            )
        ]
    )
    config = {"configurable": {"thread_id": "1"}}
    response = agent.invoke({
        "messages": [
            HumanMessage(content="请帮我查询今天北京的天气"
                                 "查询今日新闻"
                                 "查看ID位'sk131421'的邮件内容"
                                 "向15641685664@qq.com发送邮件，标题是'哈哈哈'，内容是：'你好啊'"
                                 "同时做这四件事"
                         )
        ]
    },
        config=config)
    print("========第一次invoke返回========")
    print("============原始响应============")
    rprint(response)
    print("========美化输出=========")
    for msg in response["messages"]:
        msg.pretty_print()
    interrupts = response.get("__interrupt__", [])
    print("=========interrupt===========")
    rprint(interrupts)
    print("===========开始审批============")
    weather_decision = {
        "type": "edit",
        "edited_action": {
            "name": "get_weather",
            "args": {"city": "中国北京", "is_forcast": True}
        }
    }
    news_decision = {
        "type": "approve"
    }
    send_mail_decision = {
        "type": "approve"
    }
    decisions = {
        "decisions": []
    }
    action_requests = interrupts[0].value["action_requests"]
    for action_request in action_requests:
        if action_request["name"] == "get_weather":
            decisions["decisions"].append(weather_decision)
        if action_request["name"] == "get_news":
            decisions["decisions"].append(news_decision)
        if action_request["name"] == "send_email_tool":
            decisions["decisions"].append(send_mail_decision)
    if interrupts:
        resumed_response = agent.invoke(
            Command(resume=decisions),
            config=config
        )
        print("==========审批后继续执行========")
        for msg in resumed_response["messages"]:
            msg.pretty_print()


def demo_03():
    agent = create_agent(
        model=model,
        middleware=[
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
            PIIMiddleware("url", strategy="hash", apply_to_input=True),
            PIIMiddleware("mac_address", strategy="mask", apply_to_input=True),
            PIIMiddleware("ip", strategy="block", apply_to_input=True),
        ]
    )
    response = agent.invoke({
        "messages": [
            HumanMessage("""
            帮我向 156168188@qq.com 发送一封邮件
            同时查看银行卡号： 5105-1051-0510-5100 的余额
            访问 https://localhost:12345
            确认这是不是 MAC地址： 11-11-11-11-11-11
            """)
        ]
    })
    for msg in response["messages"]:
        msg.pretty_print()
    try:
        res = agent.invoke({
            "messages": [HumanMessage("看看这个 IP 能不能 ping 通：192.168.10.1")]
        })
    except Exception as e:
        print("=" * 30, "异常", "=" * 30)
        print(f"检测到IP抛出异常：{e}")


WORK_SPACE = Path("../langchain_2026")


@tool
def list_file(path: str = ".") -> str:
    """
    列出工作区指定目录下的文件和子目录。path只能是相对路径
    Args:
        path: path: 工作区下的相对路径，一定指向目录，默认为.，表示工作区根路径，不能访问工作区
外的目录
    """
    target = (WORK_SPACE / path).resolve()
    workspace_root = WORK_SPACE.resolve()
    if not str(target).startswith(str(workspace_root)):
        return "错误：只允许访问工作区内的目录"
    if not target.exists():
        return f"错误：目录 {target} 不存在"
    if not target.is_dir():
        return f"错误：{target} 不是目录"

    items = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    if not items:
        return f"目录 {target} 为空"
    lines = []
    for item in items:
        rel = item.relative_to(workspace_root)
        kind = "[DIR]" if item.is_dir() else "[FILE]"
        lines.append(f"{kind} {rel.as_posix()}")
    return "\n".join(lines)


@tool
def read_file(path: str) -> str:
    """
    读取工作区中的文本文件内容。path只能是相对路径
    Args:
        path: 工作区内的文件名
    """
    file_path = (WORK_SPACE / path).resolve()
    if not str(file_path).startswith(str(WORK_SPACE.resolve())):
        return "错误：只允许访问工作区内的文件"
    if not file_path.exists():
        return f"错误：文件 {file_path} 不存在"
    return file_path.read_text(encoding="utf-8")


@tool
def write_file(path: str, content: str) -> str:
    """
    写入工作区中文本文件。path只能是相对路径
    Args:
        path: 工作区的文件名
        content: 写入文件的内容
    """
    file_path = (WORK_SPACE / path).resolve()
    if not str(file_path).startswith(str(WORK_SPACE.resolve())):
        return "错误：只允许访问工作区内的文件"
    file_path.write_text(content, encoding="utf-8")
    return f"文件 {file_path} 写入成功"


@tool
def run_test() -> str:
    """
    在工作区运行pytest -q，并返回输出。
    不接受任何参数，返回格式为
    returncode=0|1
    STDOUT:
    STDERR:
    """
    try:
        result = subprocess.run(
            ["pytest", "-q"],
            cwd=str(WORK_SPACE),
            capture_output=True,
            text=True,
            timeout=20
        )
        return (
            f"returncode={result.returncode}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )
    except Exception as e:
        return f"运行测试时发生异常：{e}"


def demo_04():
    agent = create_agent(
        model=model,
        tools=[list_file, read_file, write_file, run_test],
        middleware=[
            TodoListMiddleware()
        ],
        system_prompt=(
            "你是一个代码修复助手。遇到多步骤任务时，先使用 write_todos 制定待办事项；"
            "然后读取文件、修复代码并运行测试。工作全部在工作区下进行。"
        )
    )
    print('正在执行agent任务...')
    final_state = agent.invoke({
        "messages": [
            HumanMessage(content="请测试并修复工作区下my_add.py文件中的代码,修复之前需要先指定代办事项然后再进行修复")
        ]
    })
    rprint(final_state)


@before_model
def before_model_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    state["messages"][-1].content += " -> before_model <-"
    return None


@after_model
def after_model_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    state["messages"][-1].content += " -> after_model <-"
    return None


@before_agent
def before_agent_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    state["messages"][-1].content += " -> before_agent <-"
    return None


@after_agent(can_jump_to=["end"])
def after_agent_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    state["messages"][-1].content += " -> after_agent <-"
    return None


def demo_05():
    agent = create_agent(
        model=model,
        middleware=[
            before_agent_middleware,
            after_agent_middleware,
            before_model_middleware,
            after_model_middleware
        ]
    )
    response = agent.invoke({
        "messages": [
            HumanMessage(content="你好啊")
        ]
    })
    for msg in response["messages"]:
        msg.pretty_print()


class MyMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        state["messages"][-1].content += " -> before_agent <-"
        return None

    @hook_config(can_jump_to=["tools", "end"])
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        text = state['messages'][-1].content
        if "overflow" in text:
            print("[MIDDLEWARE] before_model: jump_to='end' when context window overflow")
            return {
                "messages": [AIMessage("上下文窗口溢出，终止")],
                "jump_to": "end"
            }
        if isinstance(text, str) and 'direct tool' in text.lower():
            print("[MIDDLEWARE] before_model: jump_to='tools'")
            fake_tool_call = AIMessage(
                content="人工构造的消息",
                tool_calls=[
                    {
                        "name": "get_news",
                        "args": {},
                        "id": "call_force_news_001"
                    }
                ]
            )
            return {
                "messages": [fake_tool_call],
                "jump_to": "tools"
            }
        return None

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_msg = state['messages'][-1]
        if isinstance(last_msg, AIMessage) and getattr(last_msg, 'tool_calls', None):
            return None
        user_text = ""
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", "") == "human":
                user_text = getattr(msg, "content", "")
                break
        if isinstance(user_text, str) and 'retry model' in user_text.lower():
            already_injected = any(
                isinstance(getattr(msg, 'content', None), str)
                and '你必须以【二次回答】开头' in msg.content
                for msg in state['messages']
            )
            if already_injected:
                return None
            print("[MIDDLEWARE] after_model: jump_to='model' with extra system instruction")
            return {
                "messages": [
                    SystemMessage("你必须以【二次回答】开头，并且只用一句话回答。")
                ],
                "jump_to": "model"
            }
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        state["messages"][-1].content += " -> after_agent <-"
        return None


def demo_06():
    agent = create_agent(
        model=model,
        middleware=[
            MyMiddleware()
        ]
    )
    response = agent.invoke({
        "messages": [
            HumanMessage(content="你好啊")
        ]
    })
    for msg in response["messages"]:
        msg.pretty_print()


def run_once(agent, user_input):
    response = agent.invoke({
        "messages": [
            HumanMessage(content=user_input)
        ]
    })
    for msg in response["messages"]:
        msg.pretty_print()


def demo_07():
    agent = create_agent(
        model=model,
        tools=[get_news],
        middleware=[
            MyMiddleware()
        ]
    )
    print('=' * 30, '-> Case 1 <-', '=' * 30)
    run_once(agent, "请帮我查今日新闻 direct tool")
    print('=' * 30, '-> Case 2 <-', '=' * 30)
    run_once(agent, "请随便介绍一下 LangChain retry model")
    print('=' * 30, '-> Case 3 <-', '=' * 30)
    run_once(agent, "你好 overflow")
    print('=' * 30, '-> Case 4 <-', '=' * 30)
    run_once(agent, "今日新闻摘要？")


@wrap_model_call
def wrap_model_call_middleware(request: ModelRequest,
                               handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse | None:
    request.messages[-1].content += " -> wrap_model_call_before <-"
    response = handler(request)
    response.result[0].content += " -> wrap_model_call_after <-"
    return response


class WrapModelCallMiddleware(AgentMiddleware):

    def wrap_model_call(self, request: ModelRequest,
                        handler: Callable[[ModelRequest], ModelResponse]
                        ) -> ModelResponse | None:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        original_content = request.system_message.content if request.system_message else ""
        new_content = f"""{original_content}
            当前时间：{current_time}
            用户位置：中国
            语言偏好：中文
        """
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)


def demo_08():
    agent = create_agent(
        model=model,
        middleware=[
            # wrap_model_call_middleware
            WrapModelCallMiddleware()
        ]
    )
    response = agent.invoke({
        "messages": [
            HumanMessage(content="hello, what the time now ")
        ]
    })
    for msg in response["messages"]:
        msg.pretty_print()


@wrap_tool_call
def wrap_tool_call_middleware(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    result = handler(request)
    print(f"原始参数：{request.tool_call['args']}")
    print(f"原始结果：{result}")
    request.tool_call['args']['is_forcast'] = True
    result = handler(request)
    print(f"修改后参数：{request.tool_call['args']}")
    print(f"修改后结果：{result}")
    return result


class WrapToolCallMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request: ToolCallRequest,
                       handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
        result = handler(request)
        print(f"原始参数：{request.tool_call['args']}")
        print(f"原始结果：{result}")
        request.tool_call['args']['is_forcast'] = True
        result = handler(request)
        print(f"修改后参数：{request.tool_call['args']}")
        print(f"修改后结果：{result}")
        return result


def demo_09():
    agent = create_agent(
        model=model,
        tools=[get_weather],
        middleware=[
            # wrap_tool_call_middleware
            WrapToolCallMiddleware()
        ]
    )
    response = agent.invoke({
        "messages": [
            HumanMessage(content="请帮我查询今天北京的天气")
        ]
    })
    for msg in response["messages"]:
        msg.pretty_print()


if __name__ == '__main__':
    # demo_01()
    demo_09()
