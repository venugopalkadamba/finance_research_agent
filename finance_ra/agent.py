from .tools import *

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, ToolMessage

from typing_extensions import Literal
from langgraph.graph import MessagesState, StateGraph, START, END


load_dotenv()

MODEL_NAME = "llama-3.1-8b-instant"
# MODEL_NAME = "gpt-4o-mini"


llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model=MODEL_NAME)
# llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY)

available_tools = [
    CompanyInfoRetriever(),
    DividendEarningsDateRetriever(),
    MutualFundHoldersRetriever(),
    InstitutionalHoldersRetriever(),
    StockUpgradesDowngradesRetriever(),
    StockSplitsHistoryRetriever(),
    StockNewsRetriever()
]
tools_by_name = {tool.name: tool for tool in available_tools}
llm_with_tools = llm.bind_tools(available_tools)

def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="""You are a highly intelligent and reliable **Finance Research Assistant Agent** designed to provide accurate, up-to-date, and insightful financial data. Your primary role is to retrieve, analyze, and present information on publicly traded companies, including their financial metrics, market trends, institutional and mutual fund holdings, stock upgrades/downgrades, stock splits, and news.  

Your responses must be:  
- **Accurate & Factual:** Ensure all data is retrieved correctly from reliable sources. Avoid speculation or assumptions.  
- **Clear & Concise:** Present information in a well-structured manner, avoiding unnecessary complexity.  
- **Objective & Neutral:** Report financial data without bias or subjective opinions.  
- **Context-Aware:** Understand user queries and provide the most relevant insights based on available data.  

If a user requests information beyond your capabilities, respond with transparency and suggest alternative approaches where applicable. You are a trusted research tool, not a financial advisor, and should refrain from making investment recommendations or predictions."""
                    )
                ]
                + state["messages"]
            )
        ]
    }

def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        print(f"Using: {tool_call['name']}")
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    
    return {"messages": result}

# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["environment", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "Action"
    # Otherwise, we stop (reply to the user)
    return END


def init_agent():
    agent_builder = StateGraph(MessagesState)

    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("environment", tool_node)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {
            "Action": "environment",
            END: END,
        },
    )
    agent_builder.add_edge("environment", "llm_call")

    agent = agent_builder.compile()

    return agent

# agent = init_agent()

# init_assistant_message = AIMessage(content="I am an Intelligent Finance Research Assistant! How can I help you today?")
# init_assistant_message.pretty_print()

# human_message = ""
# messages = []
# while True:
#     human_message = str(input())

#     if human_message == "STOP":
#         break

#     messages.append(HumanMessage(content=human_message))

#     messages = agent.invoke({"messages": messages})

#     messages["messages"][-1].pretty_print()

#     messages = messages["messages"]