# tools.py
import os
from langchain_community.tools import GoogleSearchRun
from langchain_community.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv

load_dotenv()

# Configure the search tool
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")

# Ensure the environment variables are set
if not google_api_key or not google_cse_id:
    raise ValueError("GOOGLE_API_KEY and GOOGLE_CSE_ID must be set in the .env file")

search_wrapper = GoogleSearchAPIWrapper(google_api_key=google_api_key, google_cse_id=google_cse_id)

search_tool = GoogleSearchRun(
    name="real_time_search",
    description="Search Google for recent results, news, and real-time information about a topic.",
    api_wrapper=search_wrapper
)