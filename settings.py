import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Validate required environment variables
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY environment variable is required") 