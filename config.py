"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
# Ưu tiên GEMINI_API_KEY (Anh Đạt dùng Gemini). Giữ OPENAI_API_KEY làm fallback.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# OpenAI-compat endpoint của Google — cho phép dùng openai SDK / langchain-openai
# với key Gemini mà không cần cài thêm package.
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
# gemini-3.6-flash chỉ có 20 requests/ngày ở free tier — không đủ cho pipeline
# (~120 enrichment + 40 answer + ~360 RAGAS judge calls). Dùng flash-lite:
# quota cao hơn nhiều, chất lượng vẫn đủ cho tài liệu HR/IT ngắn tiếng Việt.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Model làm judge cho RAGAS — tách riêng để có thể dùng model mạnh hơn nếu còn quota.
GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", GEMINI_MODEL)

# LLM được dùng trong pipeline (answer generation + RAGAS judge + enrichment)
LLM_PROVIDER = "gemini" if GEMINI_API_KEY else ("openai" if OPENAI_API_KEY else "none")

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# --- Devices ---
# GPU laptop (RTX 3050, 3.68 GiB) không đủ chỗ cho cả 3 model cùng lúc:
# bge-m3 retrieval (~2.2GB) + bge-reranker-v2-m3 (~2.2GB) + bge-m3 của RAGAS.
# Giữ retrieval encoder trên GPU (phải encode 100+ chunks), đẩy reranker và
# embeddings của RAGAS xuống CPU (chỉ vài chục cặp/câu hỏi) -> tránh CUDA OOM.
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "")          # "" = để sentence-transformers tự chọn
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu")
RAGAS_EMBEDDING_DEVICE = os.getenv("RAGAS_EMBEDDING_DEVICE", "cpu")

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- RAGAS ---
# Free tier Gemini giới hạn requests/phút; RAGAS mặc định 16 worker song song.
RAGAS_MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", "4"))

# --- Enrichment ---
# Cache kết quả enrichment theo nội dung chunk: chạy lại pipeline không đốt quota.
ENRICHMENT_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".cache", "enrichment.json")

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
