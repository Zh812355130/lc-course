from dataclasses import dataclass
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from typing import Callable


@dataclass
class Context:
    model: str


@wrap_model_call
def configurable_model(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    model_name = request.runtime.context.model
    model = init_chat_model(model_name)
    return handler(request.override(molde=model))


agent = create_deep_agent(model="deepseek-chat", middleware=[configurable_model], context_schema=Context)
# 动态切换模型
result = agent.invoke(
    {"messages": [{"role": "user", "content": "hello"}]},
    context=Context(model="openai:gpt-5.4")
)
