import os
from pathlib import Path
from functools import lru_cache
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

VECTOR_DIR = os.getenv("VECTOR_DIR", "./chroma_data")

@lru_cache(maxsize=1)
def get_vectorstore():
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    return Chroma(collection_name="shopper_documents", embedding_function=embeddings,
                  persist_directory=VECTOR_DIR)

def ingest_text(text: str, source: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents([Document(page_content=text, metadata={"source": source})])
    if not chunks:
        return 0
    get_vectorstore().add_documents(chunks)
    return len(chunks)

def retrieve_documents(question: str, k: int = 4) -> list[Document]:
    return get_vectorstore().similarity_search(question, k=k)

def delete_document_vectors(source: str) -> None:
    get_vectorstore().delete(where={"source": source})

def retrieve_context(question: str, k: int = 4) -> str:
    docs = retrieve_documents(question, k=k)
    if not docs:
        return "No relevant uploaded document content was found."
    return "\n\n".join(f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in docs)
