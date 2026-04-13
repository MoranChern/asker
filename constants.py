# -*- coding: utf-8 -*-
"""Project constants.

Edit this file directly (per requirement). Keep secrets out of version control in real projects.
"""

# -------------------------
# Neo4j connection
# -------------------------
NEO4J_HOST = "localhost"
NEO4J_BOLT_PORT = 58287  # e.g. docker-compose: '58287:7687'
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "1kcsy2C7Vrn9JHuh"
NEO4J_DATABASE = "neo4j"
NEO4J_URI = f"bolt://{NEO4J_HOST}:{NEO4J_BOLT_PORT}"

# -------------------------
# Safety / guardrails
# -------------------------
FORBIDDEN_CYPHER_KEYWORDS = [
    "CREATE", "MERGE", "DELETE", "DETACH", "SET", "DROP", "REMOVE",
    "CALL apoc.", "LOAD CSV", "FOREACH", "GRANT", "REVOKE",
]

# -------------------------
# Qwen / llama.cpp runtime config
# -------------------------
# GPU selection for llama-cpp (set to None to not set env var)
CUDA_VISIBLE_DEVICES = "2,3"

# Model path (GGUF)
GENERAL_MODEL_PATH = "/models/Qwen3-32B-GGUF/Qwen3-32B-Q6_K.gguf"

# Context length & GPU layers
N_CTX = 40960
N_GPU_LAYERS = -1

# -------------------------
# Qwen generation params
# -------------------------
QWEN_MAX_TOKENS = 32768
QWEN_TEMPERATURE = 0.6
QWEN_TOP_K = 20
QWEN_TOP_P = 0.95
QWEN_MIN_P = 0.0
QWEN_PRESENCE_PENALTY = 1.5

# -------------------------
# Local chat history storage
# -------------------------
# Directory used to store local chat/session context on disk.
# It can be a relative path (relative to server.py) or an absolute path.
CHAT_STORAGE_DIR = "local_context"
CHAT_DB_FILENAME = "chat_history.sqlite3"

# -------------------------
# Web server (FastAPI/Uvicorn) config
# -------------------------
# IMPORTANT: Per requirement, server.py MUST NOT read configuration from environment variables.
#           Edit these constants directly.
SERVER_BIND_HOST = "0.0.0.0"
SERVER_BIND_PORT = 8000

# Max concurrent GPU worker processes (avoid GPU OOM)
SERVER_GPU_WORKERS = 1

# GPU worker process management
SERVER_WORKER_HEARTBEAT_SEC = 2.0
SERVER_WORKER_TERMINATE_TIMEOUT_SEC = 2.0
SERVER_WORKER_KILL_TIMEOUT_SEC = 2.0

# -------------------------
# Server retrieval config (no index)
# -------------------------
SERVER_TOP_K = 8
SERVER_EXPAND_K = 20
