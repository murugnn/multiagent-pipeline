import os
import operator
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List, Optional
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig

class CustomerSupportAgent:
    def resolve_customer_issue(self, problem_description: str, product_name: str):
        print(f"--- CustomerSupportAgent: Resolving issue for '{product_name}' ---")
        return f"We are sorry you're having an issue with '{product_name}'. Here are the steps to resolve '{problem_description}'..."

class CustomerSupportSchema(BaseModel):
    problem_description: str = Field(description="A detailed description of the customer's issue.")
    product_name: str = Field(description="The name of the product the customer is having an issue with.")

customer_agent = CustomerSupportAgent()

@tool(args_schema=CustomerSupportSchema)
def customer_support_tool(problem_description: str, product_name: str) -> str:
    """Assists users with problems related to a specific product or service."""
    if not product_name:
        return "I can help with that, but I need a little more information. What is the name of the product you are using?"
    return customer_agent.resolve_customer_issue(problem_description, product_name)

tools = [customer_support_tool]

class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    
    user_info: Optional[str]
    remaining_steps: int

def summarization_node(state: State) -> dict:
    """
    A hook that summarizes the conversation history if it gets too long.
    This helps manage the context window and keeps the agent focused.
    """
    print("---")
    messages = state['messages']
    if len(messages) > 6:
        print("---SUMMARIZING HISTORY---")
        summarization_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="Concisely summarize the above conversation.")
        ])
        summarizer_llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")
        summary_chain = summarization_prompt | summarizer_llm
        summary_message = summary_chain.invoke({"chat_history": messages[:-1]})
        new_messages = [summary_message, messages[-1]]
        return {"messages": new_messages}
    return {}

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama3-70b-8192"
model = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)
checkpointer = InMemorySaver()


graph = create_react_agent(
    model,
    tools,
    state_schema=State,
    pre_model_hook=summarization_node,
    checkpointer=checkpointer,
)

if __name__ == '__main__':
    print("ReAct Agent is ready. Start a conversation.")
    conversation_id = "my-first-conversation"
    
    while True:
        user_prompt = input("You: ")
        if user_prompt.lower() in ['exit', 'quit']:
            print("Ending conversation.")
            break
        
        inputs = {"messages": [HumanMessage(content=user_prompt)]}
        config = RunnableConfig(configurable={"thread_id": conversation_id})
        
        response_stream = graph.stream(inputs, config=config)
        
        print("Agent:", end="", flush=True)
        for chunk in response_stream:
            if "agent" in chunk:
                agent_response = chunk["agent"]["messages"][-1]
                if agent_response.content:
                    print(agent_response.content, end="", flush=True)
        print("\n" + "="*50)