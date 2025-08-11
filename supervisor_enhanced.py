import os
import operator
import uuid
import asyncio
import json
import aiohttp
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Optional, Dict, Any

# --- Core LangChain and LangGraph Imports ---
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# --- MCP Client Imports (using mcp package) ---
try:
    import mcp
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    MCP_AVAILABLE = True
except ImportError:
    print(" MCP package not available. Install with: pip install mcp")
    MCP_AVAILABLE = False

# --- Load Environment Variables ---
load_dotenv()

# --- Global Configurations ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

# SPEED OPTIMIZATION: Use a small, fast model for routing and a large, powerful model for generation.
ROUTER_MODEL = "llama3-8b-8192"
GENERATION_MODEL = "llama3-70b-8192"


class FastNewsClient:
    """High-speed news client with MCP-style architecture but using direct APIs"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize HTTP session for fast requests"""
        if self._initialized:
            return
            
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5),  # 5 second timeout for speed
            connector=aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
        )
        self._initialized = True
        print("Fast News Client initialized")
    
    async def search_news_newsapi(self, query: str, max_results: int = 5) -> str:
        """Search news using NewsAPI - very fast and reliable"""
        if not NEWS_API_KEY:
            return "NewsAPI key not configured"
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'sortBy': 'publishedAt',
                'pageSize': max_results,
                'apiKey': NEWS_API_KEY,
                'language': 'en'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get('articles', [])
                    
                    news_summary = f" Latest News for '{query}':\n\n"
                    for i, article in enumerate(articles[:max_results], 1):
                        title = article.get('title', 'No title')
                        description = article.get('description', 'No description')
                        source = article.get('source', {}).get('name', 'Unknown')
                        url = article.get('url', '')
                        
                        news_summary += f"{i}. **{title}** ({source})\n   {description}\n   🔗 {url}\n\n"
                    
                    return news_summary
                else:
                    return f"NewsAPI error: {response.status}"
                    
        except Exception as e:
            return f"NewsAPI request failed: {str(e)}"
    
    async def search_brave(self, query: str, max_results: int = 5) -> str:
        """Search using Brave Search API - very fast"""
        if not BRAVE_API_KEY:
            return "Brave API key not configured"
        
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                'Accept': 'application/json',
                'X-Subscription-Token': BRAVE_API_KEY
            }
            params = {
                'q': f"{query} news latest",
                'count': max_results,
                'freshness': 'pd' 
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    web_results = data.get('web', {}).get('results', [])
                    
                    search_summary = f"🔍 Latest Search Results for '{query}':\n\n"
                    for i, result in enumerate(web_results[:max_results], 1):
                        title = result.get('title', 'No title')
                        description = result.get('description', 'No description')
                        url = result.get('url', '')
                        
                        search_summary += f"{i}. **{title}**\n   {description}\n   🔗 {url}\n\n"
                    
                    return search_summary
                else:
                    return f"Brave Search error: {response.status}"
                    
        except Exception as e:
            return f"Brave Search request failed: {str(e)}"
    
    async def search_news(self, query: str, max_results: int = 5) -> str:
        """Fast news search with fallback strategy"""
        if not self._initialized:
            await self.initialize()
        
        # Try NewsAPI first (usually fastest for news)
        if NEWS_API_KEY:
            print(" Searching NewsAPI...")
            result = await self.search_news_newsapi(query, max_results)
            if "error" not in result.lower() and "failed" not in result.lower():
                return result
        
        # Fallback to Brave Search
        if BRAVE_API_KEY:
            print("Fallback to Brave Search...")
            result = await self.search_brave(query, max_results)
            if "error" not in result.lower() and "failed" not in result.lower():
                return result
        
        # Last resort: return a message about configuration
        return ("No news APIs configured. Please add NEWS_API_KEY or BRAVE_API_KEY to your .env file.\n" 
               "- NewsAPI: https://newsapi.org/\n" 
               "- Brave Search: https://api.search.brave.com/")
    
    async def get_real_time_data(self, topic: str, data_type: str = "general") -> str:
        """Get real-time data with topic-specific optimization"""
        
        # Customize search based on data type for better results
        if data_type == "health":
            query = f"{topic} medical research study latest findings"
        elif data_type == "tech":
            query = f"{topic} technology breakthrough innovation latest"
        elif data_type == "finance":
            query = f"{topic} financial market news stock price latest"
        elif data_type == "science":
            query = f"{topic} scientific discovery research latest"
        else:
            query = f"{topic} latest news recent updates"
        
        return await self.search_news(query, 3)
    
    async def close(self):
        """Clean up HTTP session"""
        if self.session:
            await self.session.close()
            print(" News client connections closed")

# Global news client instance
news_client = FastNewsClient()


async def run_content_generation(query: str, content_type: str, context: str | None) -> str:
    """Content generation with fast real-time data"""
    llm = ChatGroq(temperature=0.7, model_name=GENERATION_MODEL)
    
    # Get real-time context if keywords suggest recent information
    if not context and any(keyword in query.lower() for keyword in ["latest", "recent", "news", "current", "today", "2024", "2025"]):
        print(" Fetching real-time content data...")
        context = await news_client.search_news(query)
    
    context_prompt = ""
    if context:
        context_prompt = f"You MUST use the following real-time context:\n<context>\n{context}\n</context>\n\n"
    
    prompt_template_str = (context_prompt + "Create a high-quality '{content_type}' about: \"{query}\". "
                          "Make it engaging, informative, and based on the most current information available.")
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    chain = prompt | llm
    return chain.invoke({"query": query, "content_type": content_type}).content.strip()

async def run_healthcare_info(health_query: str, context: str | None) -> str:
    """Healthcare information with latest medical research"""
    llm = ChatGroq(temperature=0.2, model_name=GENERATION_MODEL)
    
    # Get real-time medical context
    if not context and any(keyword in health_query.lower() for keyword in ["latest study", "recent findings", "medical news", "research", "clinical trial"]):
        print("Fetching latest medical research...")
        context = await news_client.get_real_time_data(health_query, "health")
    
    context_prompt = ""
    if context:
        context_prompt = f"Base your answer on this recent medical information:\n<context>\n{context}\n</context>\n\n"
    
    prompt_string = (
        "** Medical Disclaimer: I am an AI assistant, not a medical professional. This information is for educational purposes only. Always consult a qualified healthcare provider for medical advice.**\n\n"
        + context_prompt +
        "Provide a comprehensive, evidence-based overview of: \"{health_query}\"\n\n"
        "Include recent research findings if available, but emphasize the importance of professional medical consultation."
    )
    prompt = ChatPromptTemplate.from_template(prompt_string)
    chain = prompt | llm
    return chain.invoke({"health_query": health_query}).content.strip()

async def run_tutor_logic(topic: str, mode: str, context: str | None) -> str:
    """Educational content with latest discoveries"""
    llm = ChatGroq(temperature=0.7, model_name=GENERATION_MODEL, max_tokens=600)
    
    # Get real-time educational context
    if not context and any(keyword in topic.lower() for keyword in ["latest discovery", "recent breakthrough", "new research", "innovation"]):
        print("🎓 Fetching latest educational content...")
        context = await news_client.get_real_time_data(topic, "science")
    
    context_prompt = ""
    if context:
        context_prompt = f"Incorporate this latest information in your response:\n<context>\n{context}\n</context>\n\n"
    
    if mode == 'teach':
        prompt_string = (context_prompt + 
                        "Explain '{topic}' clearly to a high school student. "
                        "Use analogies and real-world examples. Include recent developments. "
                        "End with an engaging question to test understanding.")
    else:  # quiz mode
        prompt_string = (context_prompt + 
                        "Create a 3-question multiple-choice quiz about '{topic}'. "
                        "Include recent developments and discoveries. Provide detailed answer explanations.")
    
    prompt = ChatPromptTemplate.from_template(prompt_string)
    chain = prompt | llm
    return chain.invoke({"topic": topic}).content.strip()

def run_customer_support(problem_description: str, product_name: str) -> str:
    """Fast customer support responses"""
    llm = ChatGroq(temperature=0.5, model_name=GENERATION_MODEL, max_tokens=400)
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful customer support agent for '{product_name}'. "
        "Show empathy, provide a clear solution, and maintain a friendly tone.\n\n"
        "Customer Issue: \"{problem_description}\"\n\n"
        "If you cannot solve it, politely direct them to: support@{product_name}.com"
    )
    chain = prompt | llm
    return chain.invoke({"problem_description": problem_description, "product_name": product_name}).content.strip()

def run_marketing_pitch(user_need: str, product_name: str, product_description: str) -> str:
    """Compelling marketing content"""
    llm = ChatGroq(temperature=0.8, model_name=GENERATION_MODEL, max_tokens=300)
    prompt = ChatPromptTemplate.from_template(
        "Create a persuasive marketing message for '{product_name}' that addresses: \"{user_need}\"\n\n"
        "Product: {product_description}\n\n"
        "Make it compelling, benefit-focused, and under 150 words. Include a strong call-to-action."
    )
    chain = prompt | llm
    return chain.invoke({"user_need": user_need, "product_name": product_name, "product_description": product_description}).content.strip()


def run_async(coro):
    """Helper to run async functions in sync context"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new event loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

@tool
def content_creator_task(query: str, content_type: str = 'article') -> str:
    """Create content with real-time data integration. Supports: article, blog, summary, story, report."""
    return run_async(run_content_generation(query, content_type, None))

@tool
def healthcare_task(health_query: str) -> str:
    """Provide healthcare information with latest medical research. Always includes medical disclaimers."""
    return run_async(run_healthcare_info(health_query, None))

@tool
def tutor_task(topic: str, mode: str = 'teach') -> str:
    """Educational assistance with latest discoveries. Modes: 'teach' or 'quiz'."""
    return run_async(run_tutor_logic(topic, mode, None))

@tool
def customer_support_task(problem_description: str, product_name: str) -> str:
    """Handle customer support issues with empathy and clear solutions."""
    return run_customer_support(problem_description, product_name)

@tool
def marketing_sales_task(user_need: str, product_name: str, product_description: str) -> str:
    """Create compelling marketing content that addresses specific user needs."""
    return run_marketing_pitch(user_need, product_name, product_description)

@tool
def real_time_news_search(query: str, max_results: int = 5) -> str:
    """Get the latest news and real-time information about any topic."""
    return run_async(news_client.search_news(query, max_results))

# --- Supervisor State and Graph Setup ---
class SupervisorState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    remaining_steps: int

supervisor_tools = [
    content_creator_task, 
    customer_support_task, 
    healthcare_task, 
    marketing_sales_task, 
    tutor_task,
    real_time_news_search
]

# Enhanced supervisor prompt for better task routing
supervisor_prompt = """You are a high-speed multi-task supervisor AI. Route user requests to the most appropriate specialist:

 **Available Specialists:**
- **content_creator_task**: For blogs, articles, summaries, stories, reports
- **healthcare_task**: For medical/health questions (includes disclaimers)  
- **tutor_task**: For teaching/learning (modes: 'teach' or 'quiz')
- **customer_support_task**: For customer service issues
- **marketing_sales_task**: For marketing/sales content
- **real_time_news_search**: For latest news and current events

 **Speed Priority**: Choose the most direct path. Use real_time_news_search only when explicitly asked for news.

**Auto-Detection**: Content, health, and education tasks automatically fetch real-time data when needed."""

supervisor_model = ChatGroq(
    temperature=0, 
    groq_api_key=GROQ_API_KEY, 
    model_name=ROUTER_MODEL,
    max_tokens=500
).bind(system_message=supervisor_prompt)

supervisor_graph = create_react_agent(
    supervisor_model, 
    tools=supervisor_tools, 
    state_schema=SupervisorState, 
    checkpointer=InMemorySaver()
)


async def initialize_system():
    """Initialize the system"""
    print("Initializing Supervisor...")
    await news_client.initialize()
    
    # Check API configurations
    apis_configured = []
    if NEWS_API_KEY:
        apis_configured.append("NewsAPI")
    if BRAVE_API_KEY:
        apis_configured.append("Brave Search")
    
    if apis_configured:
        print(f"News APIs available: {', '.join(apis_configured)}")
    else:
        print(" No news APIs configured - limited real-time data")
    
    print(f" Using '{ROUTER_MODEL}' for routing and '{GENERATION_MODEL}' for generation")
    print("System ready!\n")

async def cleanup_system():
    """Clean up resources"""
    print("\ Cleaning up...")
    await news_client.close()
    print("Cleanup complete.")

async def main():
    """Main async execution function"""
    if not GROQ_API_KEY:
        print(" ERROR: GROQ_API_KEY must be set in .env file.")
        return
    
    await initialize_system()
    
    conversation_id = "fast-news-thread-v1"
    config = RunnableConfig(configurable={"thread_id": conversation_id})
    
    print(" Chat started! Type 'exit' or 'quit' to end.\n")
    print("Try asking for:")
    print("   - Latest news about AI")
    print("   - Write a blog about renewable energy")  
    print("   - Teach me about quantum computing")
    print("   - Latest medical research on diabetes\n")
    
    try:
        while True:
            user_prompt = input("You: ").strip()
            if user_prompt.lower() in ['exit', 'quit', '']:
                print("Goodbye!")
                break
            
            inputs = {"messages": [HumanMessage(content=user_prompt)]}
            
            print("\nSupervisor: ", end="", flush=True)
            final_answer = "Sorry, I couldn't process that request."
            
            try:
                # Stream and get final response
                final_chunk = None
                for chunk in supervisor_graph.stream(inputs, config=config, stream_mode="values"):
                    final_chunk = chunk
                
                # Extract the final answer
                if final_chunk and "messages" in final_chunk:
                    last_message = final_chunk["messages"][-1]
                    if isinstance(last_message, ToolMessage):
                        final_answer = last_message.content
                    elif hasattr(last_message, 'content'):
                        final_answer = last_message.content
                
            except Exception as e:
                final_answer = f"Error processing request: {str(e)}"
            
            print(final_answer)
            print("\n" + "="*80)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
    finally:
        await cleanup_system()

if __name__ == '__main__':
    # Environment check
    missing_vars = []
    if not GROQ_API_KEY:
        missing_vars.append("GROQ_API_KEY")
    
    if missing_vars:
        print(f" Missing required environment variables: {', '.join(missing_vars)}")
        print("\nCreate a .env file with:")
        print("GROQ_API_KEY=your_groq_key_here")
        print("\n Optional (for enhanced news):")
        print("NEWS_API_KEY=your_newsapi_key  # Get free at https://newsapi.org/")
        print("BRAVE_API_KEY=your_brave_key   # Get at https://api.search.brave.com/")
    else:
        # Run the main async function
        asyncio.run(main())
