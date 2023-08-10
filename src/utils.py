"""Utility Helper Functions
"""
import glob
import json
import os
from datetime import datetime
from shutil import rmtree
from typing import Callable, Dict, List, Optional

import pinecone
# from chromadb.config import Settings
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
from langchain.vectorstores import FAISS, Chroma, Pinecone, VectorStore

from config import LOGGER, MAIN_DIR


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
    ".csv": (CSVLoader, {"encoding": "ISO-8859-1"}),
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
    embeddings: Callable,
    source_directory: Optional[str] = None,
    output_directory: str = "./vectorstore",
    emb_store_type: str = "faiss",
    chunk_size: int = 1000,
    chunk_overlap: int = 250,
    exclude_pages: Optional[Dict] = None,
    pinecone_idx_name: Optional[str] = None,
    additional_docs: Optional[str] = None,
    concatenate_rows: bool = True,
    key_path: Optional[str] = os.path.join(MAIN_DIR, "auth", "api_keys.json"),
) -> VectorStore:
    """Generate New Vector Index Database

    Args:
        source_directory (str): Directory contains source documents
        embeddings (Callable): Function to convert text to vector embeddings
        output_directory (str, optional): Output directory of vector index database. Defaults to "./vectorstore".
        emb_store_type (str, optional): Type of vector index database. Defaults to "faiss".
        chunk_size (int, optional): Maximum size of text chunks (characters) after split. Defaults to 1000.
        chunk_overlap (int, optional): Maximum overlapping window between text chunks. Defaults to 250.
        exclude_pages (Optional[Dict], optional): Dictionary of pages to be excluded from documents. Defaults to None.
        pinecone_idx_name (Optional[str], optional): Name of pinecone index to be created or loaded. Defaults to None.
        additional_docs (Optional[str], optional): Additional Tables, Images or Json to be added to doc list. Defaults to None.
        key_path (Optional[str], optional): Path to file containing API info.
            Defaults to os.path.join(MAIN_DIR, "auth", "api_keys.json").

    Returns:
        Vectorstore: Vector Database
    """

    if os.path.exists(output_directory):
        rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)

    if source_directory:
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
    else:
        texts = []

    if additional_docs:
        with open(additional_docs, "r") as f:
            add_doc_infos = json.load(f)
        for add_doc_info in add_doc_infos:
            if add_doc_info["mode"] == "table":
                texts.extend(
                    convert_csv_to_documents(
                        add_doc_info, concatenate_rows=concatenate_rows
                    )
                )
            elif add_doc_info["mode"] == "json":
                texts.extend(convert_json_to_documents(add_doc_info))
            else:
                LOGGER.warning(
                    "Invalid document type. No texts added to documents list"
                )

    LOGGER.info(
        f"Total number of text chunks to create vector index store: {len(texts)}"
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
        assert "index.faiss" in os.listdir(
            output_directory
        ) and "index.pkl" in os.listdir(output_directory)

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

    LOGGER.info(
        f"Successfully created {emb_store_type} vectorstore at {output_directory}"
    )

    return db


def convert_csv_to_documents(
    table_info: Dict, concatenate_rows: bool = True
) -> List[Document]:
    """Convert a dictionary containing table information into list of Documents

    Args:
        table_info (Dict): Dictionary containing .csv table information

    Returns:
        List[Document]: List of rows inside the table
    """
    assert table_info["mode"] == "table" and table_info["filename"].endswith(".csv")
    rows = load_single_document(table_info["filename"])
    documents = []
    table_content = table_info["description"] + "\n\n"
    for row in rows:
        if concatenate_rows:
            table_content += row.page_content + "\n\n"
            table_doc = Document(
                page_content=table_content, metadata=table_info["metadata"]
            )
        else:
            row_no = row.metadata["row"]
            metadata = {k: v for k, v in table_info["metadata"].items()}
            metadata["row"] = row_no
            metadata["modal"] = table_info["mode"]
            row.page_content = table_info["description"] + ":" + row.page_content
            row.metadata = metadata
            documents.append(row)

    if concatenate_rows:
        documents.append(table_doc)

    return documents


def convert_json_to_documents(json_info: Dict) -> List[Document]:
    """Convert a dictionary containing json information into list of Documents

    Args:
        table_info (Dict): Dictionary containing .json table information

    Returns:
        List[Document]: List of Documents
    """
    return []
