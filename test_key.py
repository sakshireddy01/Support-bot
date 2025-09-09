import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

key = os.getenv("OPENAI_API_KEY", "")
print("Loaded key prefix:", key[:10], "(length:", len(key), ")")

