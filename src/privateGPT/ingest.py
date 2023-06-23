"""Script to generate embeddings store
"""
import argparse

from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from config import DOCUMENT_SOURCE, EMBSTORE_DICT, LOGGER, PGPT_EMBEDDINGS_MODEL
from utils import load_documents


def get_argument_parser():
    """Argument Parser

    Returns:
        args: argument dictionary
    """
    parser = argparse.ArgumentParser("Embedding Store Creation")
    parser.add_argument(
        "--embed_store", "-e", type=str, default="chroma", help="chroma|faiss"
    )
    parser.add_argument(
        "--inputs",
        "-i",
        type=str,
        default=DOCUMENT_SOURCE,
        help="path to document source folder",
    )
    parser.add_argument(
        "--outputs",
        "-o",
        type=str,
        default=EMBSTORE_DICT["chroma"],
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
        default=500,
        help="chunk size to split documents",
    )
    parser.add_argument(
        "--chunk_overlap",
        "-v",
        type=int,
        default=50,
        help="overlap size between chunks",
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
    embeddings_model_name = args.model
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # Load documents and split in chunks
    LOGGER.info(f"Loading documents from {source_directory}")

    documents = load_documents(source_directory)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    LOGGER.info(f"Loaded {len(documents)} documents from {source_directory}")
    LOGGER.info(
        f"Split into {len(texts)} chunks of text (max. {chunk_size} characters each)"
    )

    # Create embeddings

    # Create and store locally vectorstore
    if emb_store_type == "chroma":
        chroma_settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=emb_directory,
            anonymized_telemetry=False,
        )
        db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=emb_directory,
            client_settings=chroma_settings,
        )
        db.persist()
        db = None
    LOGGER.info(f"Completed creating {emb_store_type} embedding store.")


if __name__ == "__main__":
    main()
