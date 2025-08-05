import random
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama3-70b-8192"

class TutorAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def tutor_student(self, topic: str, mode: str = 'teach') -> str:
        """Teaches a subject or quizzes the user on it. Modes: 'teach', 'quiz'."""
        if not topic:
            return "[Error] No topic found. Please provide a subject to learn about."

        if mode == 'teach':
            prompt_string = (
                "You are an engaging and knowledgeable AI Tutor. Your goal is to explain topics simply.\n"
                "Explain the following topic to a high-school student. Use an analogy if it helps.\n"
                "Break down complex ideas into simple, digestible points. Keep the explanation under 250 words.\n"
                "End with a simple question to check for understanding.\n\n"
                "Topic to Explain:\n\"{topic}\""
            )
        elif mode == 'quiz':
            prompt_string = (
                "You are a friendly quiz master. Create a short, 3-question multiple-choice quiz about the given topic.\n"
                "The questions should test fundamental knowledge. Make one question easy, one medium, and one slightly harder.\n"
                "After the questions, provide a separate answer key with a brief explanation for each correct answer.\n\n"
                "Quiz Topic:\n\"{topic}\""
            )
        else:
            return f"[Error] Invalid mode '{mode}'. Please use 'teach' or 'quiz'."

        prompt = ChatPromptTemplate.from_template(prompt_string)
        chain = prompt | self.llm_client

        try:
            response = chain.invoke({"topic": topic})
            return response.content.strip()
        except Exception as e:
            error_msgs = [
                "I'm sorry, I'm having a bit of trouble recalling that information. Could you try another topic?",
                f"An unexpected error occurred: {str(e)}"
            ]
            return random.choice(error_msgs)

def initialize_groq_client():
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

if __name__ == '__main__':
    print("Welcome to the AI Tutor Agent")
    print("You can choose to either learn about a topic or take a quiz.\n")

    topic = input("📚 Enter the topic you want to learn or be quizzed on: ").strip()
    mode = input("🧠 Choose mode ('teach' or 'quiz'): ").strip().lower()

    groq_client = initialize_groq_client()
    if groq_client:
        tutor = TutorAgent(llm_client=groq_client)
        result = tutor.tutor_student(topic=topic, mode=mode)
        print("\n📘 Output:\n")
        print(result)
