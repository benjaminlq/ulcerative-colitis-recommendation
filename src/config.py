"""Config file
"""
import logging
import os
import sys

MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(MAIN_DIR, "data")
EMBSTORE_DIR = os.path.join(DATA_DIR, "emb_store")
ARTIFACT_DIR = os.path.join(MAIN_DIR, "artifacts")

MODEL_DIR = os.path.join(MAIN_DIR, "models")
PROMPT_DIR = os.path.join(MAIN_DIR, "prompt")
DOCUMENT_SOURCE = os.path.join(DATA_DIR, "document_store")

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


# Path to Embeddings Store
EMBSTORE_DICT = {
    "polyp": {"faiss": os.path.join(EMBSTORE_DIR, "polyp", "faiss", "v1")},
    "uc": {"faiss": os.path.join(EMBSTORE_DIR, "uc", "faiss")},
}

VALIDATION_SET = {
    "polyp": {"queries": os.path.join(DATA_DIR, "queries", "polyp.txt")},
    "uc": {
        "queries": os.path.join(DATA_DIR, "queries", "uc.txt"),
        "ground_truth": os.path.join(DATA_DIR, "queries", "uc_gt.csv"),
    },
}

# Exclude Pages
EXCLUDE_DICT = {
    "agrawal.pdf": [13, 14, 15, 16, 17, 18],
    "PIIS1542356520300446.pdf": [12, 13, 14, 15, 16, 17, 18],
    "gutjnl-2021-326390R2 CLEAN.pdf": [
        0,
        2,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
    ]
    + list(range(3, 31)),
    "otad009.pdf": [15, 16],
    "1-s2.0-S2468125321003770-main.pdf": [9],
    "juillerat 2022.pdf": [6, 7, 8],
}
