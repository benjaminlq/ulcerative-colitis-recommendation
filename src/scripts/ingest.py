"""Script to generate embeddings store
"""
import argparse
import json
import os

import openai
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings

from config import EMBSTORE_DIR, EXCLUDE_DICT, LOGGER, MAIN_DIR, PGPT_EMBEDDINGS_MODEL
from utils import generate_vectorstore


def get_argument_parser():
    """Argument Parser

    Returns:
        args: argument dictionary
    """
    parser = argparse.ArgumentParser("Embedding Store Creation")
    parser.add_argument(
        "--embed_store", "-e", type=str, default="chroma", help="chroma|faiss|pinecone"
    )
    parser.add_argument(
        "--inputs",
        "-i",
        default=None,
        type=str,
        help="path to document source folder",
    )
    parser.add_argument("--project", "-j", type=str, default=None, help="project")
    parser.add_argument(
        "--outputs",
        "-o",
        default=None,
        type=str,
        help="output directory to store embeddings",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=PGPT_EMBEDDINGS_MODEL,
        help="path to embedding model",
    )
    parser.add_argument(
        "--chunk_size",
        "-s",
        type=int,
        default=1000,
        help="chunk size to split documents",
    )
    parser.add_argument(
        "--chunk_overlap",
        "-v",
        type=int,
        default=50,
        help="overlap size between chunks",
    )
    parser.add_argument(
        "--pinecone_index_name",
        "-p",
        type=str,
        default=None,
        help="Name of pinecone index",
    )
    parser.add_argument(
        "--additional_docs",
        "-a",
        type=str,
        default=None,
        help="Path to additional documents",
    )
    parser.add_argument(
        "--disable_concatenate_rows", "-d", action="store_false", default=True
    )
    args = parser.parse_args()
    return args


def main():
    """Main Function"""
    # Load environment variables
    args = get_argument_parser()
    emb_store_type = args.embed_store.lower()
    emb_directory = args.outputs
    source_directory = args.inputs
    embeddings_model = args.model
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    pinecone_idx_name = args.pinecone_index_name
    project = args.project
    additional_docs = args.additional_docs
    concatenate_rows = args.disable_concatenate_rows

    if embeddings_model.lower() == "openai":
        emb_model_name = "text-embedding-ada-002"
        with open(os.path.join(MAIN_DIR, "auth", "api_keys.json"), "r") as f:
            keys = json.load(f)
        openai.api_key = keys["OPENAI_API_KEY"]
        embeddings = OpenAIEmbeddings(openai_api_key=keys["OPENAI_API_KEY"])
        LOGGER.info("Creating Vectorstore with OpenAI Embeddings")
    else:
        emb_model_name = embeddings_model.split("/")[-1]
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        LOGGER.info("Creating Vectorstore with Sentence Transformer Embeddings")

    if not emb_directory:
        print(EMBSTORE_DIR)
        print(project)
        print(emb_store_type)
        print(emb_model_name)
        parent_folder = os.path.join(
            EMBSTORE_DIR, project, emb_store_type, emb_model_name
        )
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder, exist_ok=True)
        emb_directory = os.path.join(
            parent_folder, f"v{len(os.listdir(parent_folder)) + 1}"
        )

    emb_directory = emb_directory + f"_{chunk_size}_{chunk_overlap}"

    generate_vectorstore(
        source_directory=source_directory,
        embeddings=embeddings,
        output_directory=emb_directory,
        emb_store_type=emb_store_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        exclude_pages=EXCLUDE_DICT,
        pinecone_idx_name=pinecone_idx_name,
        additional_docs=additional_docs,
        concatenate_rows=concatenate_rows,
    )


if __name__ == "__main__":
    main()

# Text & Tables: python3 ./src/scripts/ingest.py -e faiss -i data/document_store/uc -o data/emb_store/uc/faiss/text-embedding-ada-002/v2-add -j uc -m openai -s 1000 -v 200 -a data/additional_docs.json
# Text & Rows: python3 ./src/scripts/ingest.py -e faiss -i data/document_store/uc -o data/emb_store/uc/faiss/text-embedding-ada-002/v2-add -j uc
#               -m openai -s 1000 -v 200 -a data/additional_docs.json --disable_concatenate_rows
# Table only: python3 ./src/scripts/ingest.py -e faiss -o data/emb_store/uc/faiss/text-embedding-ada-002/v3-table-only -j uc -m openai -a data/additional_docs.json
