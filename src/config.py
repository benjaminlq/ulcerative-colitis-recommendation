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
}
PROMPT_DIR = os.path.join(MAIN_DIR, "prompt")

TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 512

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
