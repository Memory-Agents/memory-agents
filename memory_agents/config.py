GRAPHITI_MCP_URL = "http://localhost:8000/mcp"
BASELINE_MODEL_NAME = "gpt-4o-mini"

LONGMEMEVAL_URL_MAP = {
    "easy": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    "medium": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    "hard": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json",
}

LONGMEMEVAL_DIFFICIULTY_LEVEL = (
    "easy"  # Default difficulty level: "easy", "medium", or "hard"
)
