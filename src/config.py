"""Config file
"""
import logging
import os
import sys

MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
EMBSTORE_DIR = os.path.join(MAIN_DIR, "data", "emb_store")
EMBSTORE_DICT = {
    "faiss": os.path.join(EMBSTORE_DIR, "faiss"),
    "pinecone": os.path.join(EMBSTORE_DIR, "pinecone"),
    "chroma": os.path.join(EMBSTORE_DIR, "chroma"),
}
MODEL_DIR = os.path.join(MAIN_DIR, "models")
PROMPT_DIR = os.path.join(MAIN_DIR, "prompt")
DOCUMENT_SOURCE = os.path.join(MAIN_DIR, "data", "document_store", "polyp")

TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 512

# Private-GPT Settings
PGPT_MODEL_TYPE = "GPT4All"
PGPT_MODEL = os.path.join(MODEL_DIR, "ggml-gpt4all-j-v1.3-groovy.bin")
PGPT_EMBEDDINGS_MODEL = os.path.join(MODEL_DIR, "multi-qa-mpnet-base-dot-v1")
PGPT_MODEL_N_CTX = 1000

LOGGER = logging.getLogger(__name__)

stream_handler = logging.StreamHandler(sys.stdout)
log_folder = os.path.join(MAIN_DIR, "log")
if not os.path.exists(log_folder):
    os.makedirs(log_folder, exist_ok=True)

file_handler = logging.FileHandler(filename=os.path.join(log_folder, "logfile.log"))

formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(stream_handler)
LOGGER.addHandler(file_handler)
