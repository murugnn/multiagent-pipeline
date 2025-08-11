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
AGENT_NAME = "Marketing & Sales"

# --- Tool Definition ---
class MarketingPitchSchema(BaseModel):
    user_need: str = Field(description="The customer's stated need or problem.")
    product_name: str = Field(description="The name of the product.")
    product_description: str = Field(description="A brief description of the product.")

@tool(args_schema=MarketingPitchSchema)
def marketing_pitch_tool(user_need: str, product_name: str, product_description: str) -> str:
    """Generates a persuasive marketing message connecting a user's need to a product."""
    llm = ChatGroq(temperature=0.8, model_name=MODEL_NAME, max_tokens=300)

    prompt_string = (
        "You are a persuasive AI marketing assistant.\n"
        "Product Name: {product_name}\n"
        "Product Description: {product_description}\n"
        "Customer's Need: \"{user_need}\"\n\n"
        "Your Task: Write a short, engaging message (under 150 words). Address the customer's need directly. "
        "Focus on benefits, not just features. End with a clear call-to-action."
    )

    prompt = ChatPromptTemplate.from_template(prompt_string)
    chain = prompt | llm
    response = chain.invoke({
        "user_need": user_need, "product_name": product_name, "product_description": product_description
    })
    return response.content.strip()

# --- Agent Setup ---
tools = [marketing_pitch_tool]
model = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    remaining_steps: int  # <-- THIS LINE WAS ADDED

marketing_sales_agent = create_react_agent(
    model,
    tools,
    state_schema=AgentState,
    checkpointer=InMemorySaver(),
)

if __name__ == '__main__':
    from langchain_core.messages import HumanMessage
    print(f"Testing the {AGENT_NAME} Agent...")
    result = marketing_sales_agent.invoke({
        "messages": [HumanMessage(content="I need a tool to automate my social media posts. The product is called 'SocialScheduler' and it auto-posts to Twitter, Instagram, and Facebook.")]
    })
    print("--- Response ---")
    print(result['messages'][-1].content)