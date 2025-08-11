# content_creator.py
import os
import operator
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List, Optional
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama3-70b-8192"

class ContentCreatorSchema(BaseModel):
    query: str = Field(description="The topic or idea the content should be about.")
    content_type: str = Field(description="Type of content: 'blog', 'summary', 'story'.")
    context: Optional[str] = Field(None, description="Optional real-time context from a search tool.")

@tool(args_schema=ContentCreatorSchema)
def content_generation_tool(query: str, content_type: str = 'general', context: Optional[str] = None) -> str:
    """Generates content, using provided real-time context if available."""
    llm = ChatGroq(temperature=0.7, model_name=MODEL_NAME)
    
    # The prompt now dynamically incorporates the context
    context_prompt = ""
    if context:
        context_prompt = f"You MUST use the following real-time context to inform your response:\n<context>\n{context}\n</context>\n\n"

    prompt_template_str = (
        context_prompt +
        "You are an expert content creator. Create a '{content_type}' about the following query: \"{query}\". "
        "Ensure your output is high-quality, accurate, and reflects the provided context."
    )
    
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    chain = prompt | llm
    response = chain.invoke({"query": query, "content_type": content_type})
    return response.content.strip()

tools = [content_generation_tool]
model = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    remaining_steps: int

content_creator_agent = create_react_agent(model, tools, state_schema=AgentState, checkpointer=InMemorySaver())