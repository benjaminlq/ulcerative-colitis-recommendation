"""Utility Helper Functions
"""
import glob
import os
import pinecone
import json

from typing import Dict, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS, Pinecone
from chromadb.config import Settings

from config import LOGGER, MAIN_DIR
from shutil import rmtree
from datetime import datetime

class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if "text/html content not found in email" in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def load_single_document(file_path: str) -> Document:
    """Load a single document to text

    Args:
        file_path (str): path to document

    Returns:
        Document: Document file
    """
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, exclude_pages: Dict) -> List[Document]:
    """Load all documents inside a directory

    Args:
        source_dir (str): path to documents directory

    Returns:
        List[Document]: List of document files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )

    all_documents = []

    for file_path in all_files:
        filename = file_path.split("/")[-1]
        exclude_page_idx = exclude_pages[filename] if filename in exclude_pages else []
        pages = load_single_document(file_path)
        for page in pages:
            if page.metadata["page"] in exclude_page_idx:
                continue
            page.metadata["modal"] = "text"
            all_documents.append(page)

    return all_documents

def generate_vectorstore(
    source_directory: str,
    embeddings,
    output_directory: str = "./vectorstore",
    emb_store_type: str = "chroma",
    chunk_size: int=1000,
    chunk_overlap: int=250,
    exclude_pages: Optional[Dict] = None,
    pinecone_idx_name: Optional[str] = None,
    key_path: Optional[str] = os.path.join(MAIN_DIR, "auth", "api_keys.json")
    ):
    
    if os.path.exists(output_directory):
        rmtree(output_directory)
    os.makedirs(output_directory, exist_ok = True)

    LOGGER.info(f"Loading documents from {source_directory}")

    documents = load_documents(source_directory, exclude_pages=exclude_pages)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)

    LOGGER.info(f"Loaded {len(documents)} documents from {source_directory}")
    LOGGER.info(
        f"Split into {len(texts)} chunks of text (max. {chunk_size} characters each)"
    )

    if emb_store_type == "chroma":
        chroma_settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=output_directory,
            anonymized_telemetry=False,
        )
        db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=output_directory,
            client_settings=chroma_settings,
        )
        db.persist()

    elif emb_store_type == "faiss":
        db = FAISS.from_documents(texts, embedding=embeddings)
        db.save_local(output_directory)
        assert "index.faiss" in os.listdir(output_directory) and "index.pkl" in os.listdir(output_directory)

    elif emb_store_type == "pinecone":
        with open(key_path, "r") as f:
            keys = json.loads(f)
        PINECONE_API_KEY = keys["PINECONE_API"]["KEY"]
        PINECONE_ENV = keys["PINECONE_API"]["ENV"]

        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENV,
        )

        if not pinecone_idx_name:
            pinecone_idx_name = "index_{}".format(
                datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
            )

        if pinecone_idx_name not in pinecone.list_indexes():
            db = Pinecone.from_documents(
                texts, embedding=embeddings, index_name=pinecone_idx_name
            )

        else:
            db = Pinecone.from_existing_index(pinecone_idx_name, embeddings)
            db.add_documents(texts)
    
    LOGGER.info(f"Successfully created {emb_store_type} vectorstore at {output_directory}")

    return db