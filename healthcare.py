import os
import operator
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# --- Agent Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama3-70b-8192"
AGENT_NAME = "Healthcare Assistant"

# --- Tool Definition ---
class HealthInfoSchema(BaseModel):
    health_query: str = Field(description="The user's question about a health-related topic.")

@tool(args_schema=HealthInfoSchema)
def health_info_tool(health_query: str) -> str:
    """Provides general, educational health information. It does NOT give medical advice."""
    llm = ChatGroq(temperature=0.2, model_name=MODEL_NAME, max_tokens=400)
    
    prompt_string = (
        "**CRITICAL SAFETY INSTRUCTION: You are an AI Health Information Assistant, NOT a medical professional.**\n"
        "Your response MUST start with: 'Disclaimer: I am an AI assistant and not a medical professional. This information is for educational purposes only. Please consult a qualified healthcare provider for any medical advice or concerns.'\n\n"
        "After the disclaimer, provide a high-level, neutral, and informative overview of the user's topic.\n"
        "Your response MUST end by reinforcing the need to see a doctor.\n\n"
        "User's Health Query:\n\"{health_query}\""
    )

    prompt = ChatPromptTemplate.from_template(prompt_string)
    chain = prompt | llm
    response = chain.invoke({"health_query": health_query})
    return response.content.strip()

# --- Agent Setup ---
tools = [health_info_tool]
model = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    remaining_steps: int  # <-- THIS LINE WAS ADDED

healthcare_agent = create_react_agent(
    model,
    tools,
    state_schema=AgentState,
    checkpointer=InMemorySaver(),
)

if __name__ == '__main__':
    from langchain_core.messages import HumanMessage
    print(f"Testing the {AGENT_NAME} Agent...")
    result = healthcare_agent.invoke(
        {"messages": [HumanMessage(content="What are the common symptoms of iron deficiency?")]}
    )
    print("--- Response ---")
    print(result['messages'][-1].content)