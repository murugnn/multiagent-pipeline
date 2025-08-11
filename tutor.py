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
AGENT_NAME = "Tutor"

# --- Tool Definition ---
class TutorSchema(BaseModel):
    topic: str = Field(description="The subject or topic the user wants to learn or be quizzed on.")
    mode: str = Field(description="The mode of interaction: 'teach' or 'quiz'.")

@tool(args_schema=TutorSchema)
def tutor_tool(topic: str, mode: str = 'teach') -> str:
    """Teaches a subject or quizzes the user on it."""
    llm = ChatGroq(temperature=0.7, model_name=MODEL_NAME, max_tokens=600)

    if mode == 'teach':
        prompt_string = (
            "You are an engaging AI Tutor. Explain the following topic simply to a high-school student.\n"
            "Use an analogy and keep it under 250 words. End with a question to check understanding.\n\n"
            "Topic: \"{topic}\""
        )
    elif mode == 'quiz':
        prompt_string = (
            "You are a quiz master. Create a 3-question multiple-choice quiz about the given topic.\n"
            "After the questions, provide an answer key with explanations.\n\n"
            "Quiz Topic: \"{topic}\""
        )
    else:
        return f"[Error] Invalid mode '{mode}'. Please use 'teach' or 'quiz'."

    prompt = ChatPromptTemplate.from_template(prompt_string)
    chain = prompt | llm
    response = chain.invoke({"topic": topic})
    return response.content.strip()

# --- Agent Setup ---
tools = [tutor_tool]
model = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    remaining_steps: int  # <-- THIS LINE WAS ADDED

tutor_agent = create_react_agent(
    model,
    tools,
    state_schema=AgentState,
    checkpointer=InMemorySaver(),
)

if __name__ == '__main__':
    from langchain_core.messages import HumanMessage
    print(f"Testing the {AGENT_NAME} Agent...")
    result = tutor_agent.invoke(
        {"messages": [HumanMessage(content="Can you teach me about photosynthesis?")]}
    )
    print("--- Response ---")
    print(result['messages'][-1].content)