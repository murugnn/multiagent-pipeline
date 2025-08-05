import random
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama3-70b-8192"

class ContentGeneratorAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate_content(self, query: str, content_type: str = 'general') -> str:
        """Generates content of a specific type (e.g., blog, summary, story) about a given topic."""
        if not query:
            return "[Error] No query found. Please provide what you want content about."

        if content_type == 'blog':
            prompt_string = (
                "You are an expert content creator. Write a well-structured, engaging blog post about:\n\n\"{query}\"\n\n"
                "Keep it clear and informative. Use headings if suitable. Keep length moderate (under 300 words)."
            )
        elif content_type == 'summary':
            prompt_string = (
                "Summarize the following topic for an educated, general audience (4-6 sentences):\n\n\"{query}\"\n\n"
                "Make it concise and easy to understand."
            )
        elif content_type == 'story':
            prompt_string = (
                "Write a short, original, and creative story inspired by this prompt:\n\n\"{query}\"\n\n"
                "Make it engaging and under 200 words."
            )
        else:
            prompt_string = (
                "Generate high-quality, original content about:\n\n\"{query}\"\n\n"
                "Content should be useful, engaging, and suitable for a broad audience. Keep it under 300 words."
            )

        prompt = ChatPromptTemplate.from_template(prompt_string)
        chain = prompt | self.llm_client

        try:
            response = chain.invoke({"query": query})
            return response.content.strip()
        except Exception as e:
            error_msgs = [
                "Sorry, there was an error generating your content. Please try again.",
                f"Content generation failed: {str(e)}"
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

# Interactive CLI usage
if __name__ == '__main__':
    print("Welcome to the AI Content Generator")
    print("Choose a content type and a topic to generate original content!\n")

    query = input("What topic or idea should the content be about? ").strip()
    content_type = input("Content type ('blog', 'summary', 'story', or press Enter for 'general'): ").strip().lower()
    if not content_type:
        content_type = "general"

    groq_client = initialize_groq_client()
    if groq_client:
        generator = ContentGeneratorAgent(llm_client=groq_client)
        result = generator.generate_content(query=query, content_type=content_type)
        print("\nGenerated Content:\n")
        print(result)
