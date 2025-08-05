import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from content_creator import ContentGeneratorAgent
from customer_support import CustomerSupportAgent
from healthcare import HealthcareAssistantAgent
from market_sales import MarketingSalesAgent
from tutor import TutorAgent

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama3-70b-8192"

def initialize_groq_client():
    """Initializes and returns the ChatGroq client."""
    try:
        return ChatGroq(
            temperature=0.7,
            groq_api_key=GROQ_API_KEY,
            model_name=MODEL_NAME,
            max_tokens=600
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize Groq client: {e}")
        return None

def supervisor_agent(user_prompt: str, llm_client):
    """
    A supervisor agent that routes user prompts to the appropriate specialized agent.
    """
    prompt_template = f"""
    You are a supervisor agent. Your job is to determine which of the following agents is best suited to handle the user's request:
    - Content Generator: Creates content like blogs, summaries, or stories.
    - Customer Support: Helps users with problems with a product or service.
    - Healthcare Assistant: Provides general health information.
    - Marketing and Sales: Creates marketing pitches for products.
    - Tutor: Teaches or quizzes users on a topic.

    Based on the user's prompt, identify the best agent.
    User prompt: "{user_prompt}"
    Best agent: 
    """
    
    response = llm_client.invoke(prompt_template)
    best_agent = response.content.strip().lower()

    if "content" in best_agent:
        agent = ContentGeneratorAgent(llm_client)
        content_type = input("Enter content type (blog, summary, story): ")
        return agent.generate_content(query=user_prompt, content_type=content_type)
    elif "customer" in best_agent:
        agent = CustomerSupportAgent(llm_client)
        product_name = input("Enter product name: ")
        return agent.resolve_customer_issue(problem_description=user_prompt, product_name=product_name)
    elif "health" in best_agent:
        agent = HealthcareAssistantAgent(llm_client)
        return agent.provide_health_info(health_query=user_prompt)
    elif "market" in best_agent:
        agent = MarketingSalesAgent(llm_client)
        product_name = input("Enter product name: ")
        product_description = input("Enter product description: ")
        return agent.generate_marketing_pitch(user_need=user_prompt, product_name=product_name, product_description=product_description)
    elif "tutor" in best_agent:
        agent = TutorAgent(llm_client)
        mode = input("Enter mode (teach or quiz): ")
        return agent.tutor_student(topic=user_prompt, mode=mode)
    else:
        return "Sorry, I don't have an agent that can help with that."

if __name__ == '__main__':
    groq_client = initialize_groq_client()
    if groq_client:
        user_prompt = input("How can I help you today? ")
        response = supervisor_agent(user_prompt, groq_client)
        print(response)
