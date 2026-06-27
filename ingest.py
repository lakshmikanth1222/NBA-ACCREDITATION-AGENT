import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_DIR = "data"
CHROMA_DB_DIR = "chroma_db"

def ingest_documents():
    """Reads all PDFs from data directory, chunks them, and stores embeddings in ChromaDB."""
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found. Please create it and add PDF files.")
        return

    pdf_files = glob.glob(os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True)
    
    if not pdf_files:
        print(f"No PDF files found in '{DATA_DIR}'. Please add some documents.")
        return
        
    print(f"Loading documents...\n{len(pdf_files)} PDFs found.")
    
    documents = []
    for file_path in pdf_files:
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
        
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    
    # Initialize HuggingFace embeddings
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store in ChromaDB
    print(f"Saving to ChromaDB at '{CHROMA_DB_DIR}'...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    print("Done! ChromaDB index successfully created.")

if __name__ == "__main__":
    ingest_documents()
