import random
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama3-70b-8192"

class CustomerSupportAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def resolve_customer_issue(self, problem_description: str, product_name: str = 'our service') -> str:
        """Addresses a customer's problem by providing a solution or escalating to a human agent."""
        if not problem_description:
            return "[Error] No problem description found. Please describe your issue."

        prompt_string = (
            "You are a friendly, patient, and knowledgeable customer support representative for a company.\n"
            "Your goal is to help the user solve their problem with '{product_name}'.\n\n"
            "1. Start by acknowledging their problem and showing empathy.\n"
            "2. Provide a clear, step-by-step solution if possible.\n"
            "3. If the problem is complex or you cannot solve it, politely explain that you cannot resolve the issue "
            "and provide a generic instruction to contact the human support team via 'support@example.com'.\n"
            "4. Keep your response concise and easy to follow.\n\n"
            "Customer's Problem:\n\"{problem_description}\""
        )

        prompt = ChatPromptTemplate.from_template(prompt_string)
        chain = prompt | self.llm_client

        try:
            response = chain.invoke({
                "problem_description": problem_description,
                "product_name": product_name
            })
            return response.content.strip()
        except Exception as e:
            error_msgs = [
                "I'm sorry, I'm having trouble connecting to our systems right now. Please try again in a moment.",
                f"An unexpected error occurred: {str(e)}"
            ]
            return random.choice(error_msgs)

def initialize_groq_client():
    try:
        return ChatGroq(
            temperature=0.7,
            groq_api_key=GROQ_API_KEY,
            model_name=MODEL_NAME,
            max_tokens=400
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize Groq client: {e}")
        return None

if __name__ == '__main__':
    print("Welcome to the AI Customer Support Agent")
    print("Tell me your issue and I’ll try to help you out!\n")

    issue = input("Describe your problem: ").strip()
    product = input("What product or service is this related to? (press enter for 'our service'): ").strip()
    if not product:
        product = "our service"

    groq_client = initialize_groq_client()
    if groq_client:
        support_agent = CustomerSupportAgent(llm_client=groq_client)
        reply = support_agent.resolve_customer_issue(problem_description=issue, product_name=product)
        print("\nSupport Response:\n")
        print(reply)
