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
AGENT_NAME = "Customer Support"

# --- Tool Definition ---
class CustomerSupportSchema(BaseModel):
    problem_description: str = Field(description="A detailed description of the customer's issue.")
    product_name: str = Field(description="The name of the product or service the customer is using.")

@tool(args_schema=CustomerSupportSchema)
def customer_issue_resolver_tool(problem_description: str, product_name: str = 'our service') -> str:
    """Addresses a customer's problem by providing a solution or escalating to a human agent."""
    llm = ChatGroq(temperature=0.7, model_name=MODEL_NAME, max_tokens=400)
    
    prompt_string = (
        "You are a friendly and knowledgeable customer support representative.\n"
        "Your goal is to help the user solve their problem with '{product_name}'.\n\n"
        "1. Acknowledge their problem with empathy.\n"
        "2. Provide a clear, step-by-step solution if possible.\n"
        "3. If you cannot solve it, politely explain and provide an instruction to contact human support at 'support@example.com'.\n\n"
        "Customer's Problem:\n\"{problem_description}\""
    )

    prompt = ChatPromptTemplate.from_template(prompt_string)
    chain = prompt | llm
    response = chain.invoke({"problem_description": problem_description, "product_name": product_name})
    return response.content.strip()

# --- Agent Setup ---
tools = [customer_issue_resolver_tool]
model = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    remaining_steps: int  # <-- THIS LINE WAS ADDED

customer_support_agent = create_react_agent(
    model,
    tools,
    state_schema=AgentState,
    checkpointer=InMemorySaver(),
)

if __name__ == '__main__':
    from langchain_core.messages import HumanMessage
    print(f"Testing the {AGENT_NAME} Agent...")
    result = customer_support_agent.invoke(
        {"messages": [HumanMessage(content="My account is locked and I can't reset the password. I'm using the 'Pro Plan'.")]}
    )
    print("--- Response ---")
    print(result['messages'][-1].content)