from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()


class Person(BaseModel):
    name: str = Field(description="姓名")
    age: Optional[int] = Field(description="年龄")
    occupation: str = Field(description="职业")


class MovieModel(BaseModel):
    title: str = Field(description="电影标题")
    year: int = Field(description="电影上映年份")
    director: str = Field(description="导演")
    rating: float = Field(description="电影评分，满分10分")


def demo_01():
    model = init_chat_model(model="deepseek:deepseek-v4-flash",
                            extra_body={'thinking': {"type": "disabled"}})
    # structured_llm = model.with_structured_output(Person,include_raw=True)
    structured_llm = model.with_structured_output(Person)
    result = structured_llm.invoke("张三是一名30岁的软件工程师")
    print(result)
    print(type(result))
    print(result.name)
    print(result.age)
    print(result.occupation)


def demo_02():
    model = init_chat_model(model="deepseek:deepseek-v4-flash",
                            extra_body={'thinking': {"type": "disabled"}})
    structured_llm = model.with_structured_output(MovieModel)
    result = structured_llm.invoke("给出盗梦空间的信息")
    print(result)
    print(type(result))


if __name__ == '__main__':
    demo_02()
