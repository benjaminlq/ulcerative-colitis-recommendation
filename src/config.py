import os

MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
EMBSTORE_DIR = os.path.join(MAIN_DIR, "data", "emb_store")
EMBSTORE_DICT = {
    "faiss": os.path.join(EMBSTORE_DIR, "faiss"),
    "pinecone": os.path.join(EMBSTORE_DIR, "pinecone")
}
PROMPT_DIR = os.path.join(MAIN_DIR, "prompt")

TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 800