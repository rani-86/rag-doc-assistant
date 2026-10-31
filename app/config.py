
import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve everything from the project root so the app works no matter
# which directory it is launched from.
BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
PERSIST_DIRECTORY = DATA_DIR / "chroma_db"
FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"

COLLECTION_NAME = "technical_documentation"

# Tells the query-analysis LLM what the corpus is about, so it can write
# search queries in the corpus's vocabulary.
CORPUS_DESCRIPTION = os.getenv(
    "CORPUS_DESCRIPTION",
    "the official FastAPI (Python web framework) tutorial: path parameters, "
    "query parameters, request bodies with Pydantic models, handling errors "
    "and HTTPException, and dependency injection with Depends"
)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Main model: query analysis and answer generation.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Smaller model for binary decisions (grading, rewriting, hallucination
# check). On Groq's free tier each model has its own tokens-per-minute
# budget, so splitting the work avoids hitting rate limits.
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval and self-correction
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
MAX_GENERATIONS = int(os.getenv("MAX_GENERATIONS", "2"))

# Conversation memory: number of previous turns used to resolve follow-ups
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "3"))

# Ingestion limits
SUPPORTED_EXTENSIONS = {".md", ".txt", ".html", ".htm"}
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
