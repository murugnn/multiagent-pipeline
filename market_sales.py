import random
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama3-70b-8192"

class MarketingSalesAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate_marketing_pitch(self, user_need: str, product_name: str, product_description: str) -> str:
        """Generates a persuasive marketing message connecting a user's need to a product."""
        if not all([user_need, product_name, product_description]):
            return "[Error] Missing required information. Please provide user need, product name, and description."

        prompt_string = (
            "You are a persuasive and helpful AI marketing assistant. Your goal is to show how a product is the perfect solution for a customer's needs.\n\n"
            "**Product Information:**\n"
            "- Product Name: {product_name}\n"
            "- Product Description: {product_description}\n\n"
            "**Customer's Stated Need:**\n"
            "\"{user_need}\"\n\n"
            "**Your Task:**\n"
            "Write a short, engaging, and persuasive message (under 150 words). Address the customer's need directly. "
            "Focus on the *benefits* of the product, not just its features. Be enthusiastic but not overly aggressive. "
            "End with a clear and friendly call-to-action, like inviting them to a website or to ask another question."
        )

        prompt = ChatPromptTemplate.from_template(prompt_string)
        chain = prompt | self.llm_client

        try:
            response = chain.invoke({
                "user_need": user_need,
                "product_name": product_name,
                "product_description": product_description
            })
            return response.content.strip()
        except Exception as e:
            error_msgs = [
                "I'm sorry, our marketing-tron 5000 is on a coffee break. Please try again shortly.",
                f"An unexpected error occurred: {str(e)}"
            ]
            return random.choice(error_msgs)

def initialize_groq_client():
    try:
        return ChatGroq(
            temperature=0.8,
            groq_api_key=GROQ_API_KEY,
            model_name=MODEL_NAME,
            max_tokens=300
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize Groq client: {e}")
        return None

if __name__ == '__main__':
    print("Welcome to the AI Marketing & Sales Assistant")
    print("Let's craft a persuasive pitch for your product!\n")

    user_need = input("What is the user's need or problem? ").strip()
    product_name = input("What is the name of the product? ").strip()
    product_description = input("Describe the product briefly: ").strip()

    groq_client = initialize_groq_client()
    if groq_client:
        sales_agent = MarketingSalesAgent(llm_client=groq_client)
        pitch = sales_agent.generate_marketing_pitch(user_need=user_need, product_name=product_name, product_description=product_description)
        print("\nGenerated Marketing Pitch:\n")
        print(pitch)
