import random
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama3-70b-8192"

class HealthcareAssistantAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def provide_health_info(self, health_query: str) -> str:
        """Provides general, educational health information in response to a user's query."""
        if not health_query:
            return "[Error] No query found. Please ask a health-related question."

        prompt_string = (
            "**CRITICAL SAFETY INSTRUCTION: You are an AI Health Information Assistant, NOT a medical professional.**\n"
            "**You MUST NOT provide medical advice, diagnosis, or treatment recommendations.**\n"
            "Your sole purpose is to provide general, educational information about health topics. "
            "Your response MUST begin with the following disclaimer: 'Disclaimer: I am an AI assistant and not a medical professional. This information is for educational purposes only. Please consult a qualified healthcare provider for any medical advice or concerns.'\n\n"
            "After the disclaimer, provide a high-level, neutral, and informative overview of the user's topic. "
            "Do not use speculative language. Stick to widely accepted, general knowledge.\n"
            "Your response MUST end by reinforcing the need to see a doctor, for example: 'For personalized advice, diagnosis, or treatment, it is essential to speak with a doctor or other qualified healthcare professional.'\n\n"
            "User's Health Query:\n\"{health_query}\""
        )

        prompt = ChatPromptTemplate.from_template(prompt_string)
        chain = prompt | self.llm_client

        try:
            response = chain.invoke({"health_query": health_query})
            return response.content.strip()
        except Exception as e:
            return (
                "Disclaimer: I am an AI assistant and not a medical professional. "
                "I am currently experiencing a technical issue and cannot provide information at this time. "
                "Please consult a qualified healthcare provider for any medical advice or concerns."
            )

def initialize_groq_client():
    try:
        return ChatGroq(
            temperature=0.2,
            groq_api_key=GROQ_API_KEY,
            model_name=MODEL_NAME,
            max_tokens=400
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize Groq client: {e}")
        return None

# Interactive CLI usage
if __name__ == '__main__':
    print("Welcome to the AI Healthcare Assistant")
    print("I can provide general educational health information. Let's begin!\n")

    user_query = input("What health-related question would you like to ask? ").strip()

    groq_client = initialize_groq_client()
    if groq_client:
        health_agent = HealthcareAssistantAgent(llm_client=groq_client)
        reply = health_agent.provide_health_info(health_query=user_query)
        print("\nResponse:\n")
        print(reply)
