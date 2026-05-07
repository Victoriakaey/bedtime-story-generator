import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "").strip().lower() in {"1", "true", "yes", "y"}
MAX_REVISION_PASSES = 2
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))

EXAMPLE_REQUEST = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."
