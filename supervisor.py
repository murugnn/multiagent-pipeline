import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict
from langgraph.graph import StateGraph, END

class ContentGeneratorAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    def generate_content(self, query: str, content_type: str):
        prompt = f"You are a content generator. Create a {content_type} about the following topic: {query}"
        response = self.llm_client.invoke(prompt)
        return response.content

class CustomerSupportAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    def resolve_customer_issue(self, problem_description: str, product_name: str):
        prompt = f"You are a customer support agent for the product '{product_name}'. A customer has the following problem: '{problem_description}'. Provide a helpful and empathetic response to resolve their issue."
        response = self.llm_client.invoke(prompt)
        return response.content

class HealthcareAssistantAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    def provide_health_info(self, health_query: str):
        prompt = f"You are a helpful healthcare assistant. Provide general health information about the following query. IMPORTANT: Remind the user that you are not a real doctor and they should consult a professional for medical advice. Query: {health_query}"
        response = self.llm_client.invoke(prompt)
        return response.content

class MarketingSalesAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    def generate_marketing_pitch(self, user_need: str, product_name: str, product_description: str):
        prompt = f"You are a marketing and sales expert. A potential customer has this need: '{user_need}'. Create a compelling marketing pitch for the product '{product_name}', which is described as: '{product_description}'."
        response = self.llm_client.invoke(prompt)
        return response.content

class TutorAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    def tutor_student(self, topic: str, mode: str):
        prompt = f"You are a tutor. The student wants to learn about '{topic}'. Your mode is '{mode}'. Engage the student accordingly."
        if mode.lower() == 'quiz':
            prompt += " Ask them a question about the topic and wait for their answer."
        else:
            prompt += " Explain a key concept about the topic in a simple and clear way."
        response = self.llm_client.invoke(prompt)
        return response.content

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama3-70b-8192"

class AgentState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        user_prompt: The initial prompt from the user.
        agent_choice: The decision made by the supervisor on which agent to use.
        response: The final response from the chosen agent.
    """
    user_prompt: str
    agent_choice: str
    response: str


def supervisor_node(state: AgentState):
    """
    The supervisor node that routes user prompts to the appropriate specialized agent.
    This node updates the 'agent_choice' in the state.
    """
    print("---SUPERVISOR---")
    user_prompt = state['user_prompt']
    
    prompt_template = f"""
    You are a supervisor agent. Your job is to determine which of the following agents is best suited to handle the user's request:
    - Content Generator: Creates content like blogs, summaries, or stories.
    - Customer Support: Helps users with problems with a product or service.
    - Healthcare Assistant: Provides general health information.
    - Marketing and Sales: Creates marketing pitches for products.
    - Tutor: Teaches or quizzes users on a topic.

    Based on the user's prompt, identify the best agent. Do not add any other text, just the agent name.
    User prompt: "{user_prompt}"
    Best agent: 
    """
    
    llm_client = initialize_groq_client() 
    response = llm_client.invoke(prompt_template)
    best_agent = response.content.strip().lower()
    
    print(f"Supervisor chose: {best_agent}")
    return {"agent_choice": best_agent}

def content_generator_node(state: AgentState):
    print("---CONTENT GENERATOR---")
    llm_client = initialize_groq_client()
    agent = ContentGeneratorAgent(llm_client)
    
    content_type = input("Enter content type (blog, summary, story): ")
    response = agent.generate_content(query=state['user_prompt'], content_type=content_type)
    return {"response": response}

def customer_support_node(state: AgentState):
    print("---CUSTOMER SUPPORT---")
    llm_client = initialize_groq_client()
    agent = CustomerSupportAgent(llm_client)
    
    product_name = input("Enter product name: ")
    response = agent.resolve_customer_issue(problem_description=state['user_prompt'], product_name=product_name)
    return {"response": response}

def healthcare_assistant_node(state: AgentState):
    print("---HEALTHCARE ASSISTANT---")
    llm_client = initialize_groq_client()
    agent = HealthcareAssistantAgent(llm_client)

    response = agent.provide_health_info(health_query=state['user_prompt'])
    return {"response": response}

def marketing_sales_node(state: AgentState):
    print("---MARKETING & SALES---")
    llm_client = initialize_groq_client()
    agent = MarketingSalesAgent(llm_client)
    
    product_name = input("Enter product name: ")
    product_description = input("Enter product description: ")
    response = agent.generate_marketing_pitch(user_need=state['user_prompt'], product_name=product_name, product_description=product_description)
    return {"response": response}

def tutor_node(state: AgentState):
    print("---TUTOR---")
    llm_client = initialize_groq_client()
    agent = TutorAgent(llm_client)

    mode = input("Enter mode (teach or quiz): ")
    response = agent.tutor_student(topic=state['user_prompt'], mode=mode)
    return {"response": response}
    
def handle_error_node(state: AgentState):
    print("---ERROR HANDLER---")
    return {"response": "Sorry, I don't have an agent that can help with that."}


def router(state: AgentState):
    """
    This function inspects the 'agent_choice' in the state and returns the
    name of the next node to execute.
    """
    choice = state['agent_choice']
    if "content" in choice:
        return "content_generator"
    elif "customer" in choice:
        return "customer_support"
    elif "health" in choice:
        return "healthcare_assistant"
    elif "market" in choice or "sales" in choice:
        return "marketing_sales"
    elif "tutor" in choice:
        return "tutor"
    else:
        return "error_handler"

def initialize_groq_client():
    """Initializes and returns the ChatGroq client."""
    try:
        return ChatGroq(
            temperature=0.7,
            groq_api_key=GROQ_API_KEY,
            model_name=MODEL_NAME,
            max_tokens=800 
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize Groq client: {e}")
        return None

workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("content_generator", content_generator_node)
workflow.add_node("customer_support", customer_support_node)
workflow.add_node("healthcare_assistant", healthcare_assistant_node)
workflow.add_node("marketing_sales", marketing_sales_node)
workflow.add_node("tutor", tutor_node)
workflow.add_node("error_handler", handle_error_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    router,
    {
        "content_generator": "content_generator",
        "customer_support": "customer_support",
        "healthcare_assistant": "healthcare_assistant",
        "marketing_sales": "marketing_sales",
        "tutor": "tutor",
        "error_handler": "error_handler",
    }
)
workflow.add_edge("content_generator", END)
workflow.add_edge("customer_support", END)
workflow.add_edge("healthcare_assistant", END)
workflow.add_edge("marketing_sales", END)
workflow.add_edge("tutor", END)
workflow.add_edge("error_handler", END)

app = workflow.compile()

if __name__ == '__main__':
    while True:
        user_prompt = input("How can I help you today? (or type 'exit' to quit) ")
        if user_prompt.lower() in ['exit', 'quit']:
            break
            
        initial_state = {"user_prompt": user_prompt, "agent_choice": "", "response": ""}
        
        final_state = app.invoke(initial_state)
        
        print("\n---FINAL RESPONSE---")
        print(final_state['response'])
        print("\n" + "="*50 + "\n")