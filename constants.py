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
CUDA_VISIBLE_DEVICES = "0,1,2"

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
