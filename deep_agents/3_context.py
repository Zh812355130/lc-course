from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

base_dir = Path(__file__).parent
agent_md_path = str(Path.joinpath(base_dir, "AGENTS.md"))
skill_path = str(Path.joinpath(base_dir, "skills"))

agent = create_deep_agent(model="deepseek-chat",
                          system_prompt="你是一名高效率的AI专家，能合理规划任务，对于繁重任务使用子智能体来完成任务。",
                          memory=[agent_md_path],
                          backend=LocalShellBackend(root_dir=".", virtual_mode=True),
                          skills=[skill_path])

# result = agent.invoke({"messages": [{"role": "user", "content": "执行shell命令创建A.txt文件,并把滕王阁序的内容写进去"}]})
result = agent.invoke({"messages": [{"role": "user", "content": "讲个笑话"}]})

for message in result["messages"]:
    message.pretty_print()
