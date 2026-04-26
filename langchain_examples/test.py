from typing import Optional

from langgraph.constants import START,END
from langgraph.graph import StateGraph
from pydantic import BaseModel



class MessagesState(BaseModel):
    name: Optional[str] = None
    llm_calls: Optional[int] = None

agent_builder = StateGraph(MessagesState)

# 节点函数
def say_hello(state: MessagesState):
    print('hello')
    state.name = '张亮'
    return state


def say_hi(state: MessagesState):
    print('hi')
    return state

def say_nihao(state: MessagesState):
    print('你好')
    return state


def conditional_say(state: MessagesState):
    if state.name == '张亮':
        return 'say_nihao'
    else:
        return 'say_hi'

agent_builder.add_node('say_hello',say_hello)
agent_builder.add_node('say_hi',say_hi)
agent_builder.add_node('say_nihao',say_nihao)

agent_builder.add_edge(START, "say_hello")
agent_builder.add_edge('say_hello', "say_hi")
# agent_builder.add_edge('say_hello', "say_nihao")
# agent_builder.add_edge("say_hi", END)
agent_builder.add_conditional_edges("say_hello",conditional_say)

agent = agent_builder.compile()

agent.invoke({"messages": 'hi'})

