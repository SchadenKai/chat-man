import os
from dotenv import load_dotenv

load_dotenv()


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")
