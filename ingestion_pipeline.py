import os
from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents.base import Document
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()



def load_documents(docs_path="docs"):
    # Load all text files from the docs directory
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

# def split_documents(documents, chunk_size = 800, chunk_overlap = 0):
    # Split docs into smaller groups (called as Chunks) with overlap.
def split_documents(documents):
    print("Split docs into Semantic Chunks...")
    
    # text_splitter = CharacterTextSplitter(
    #     chunk_size = chunk_size,
    #     chunk_overlap = chunk_overlap
    # )
    
    # text_splitter = RecursiveCharacterTextSplitter(
    # chunk_size=1500,
    # chunk_overlap=200,
    # separators=["\n\n", "\n", " ", ""]
    # )

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    text_splitter = SemanticChunker(
        embeddings = embedding_model,
        breakpoint_threshold_type = "percentile",
        breakpoint_threshold_amount = 95
    )

    chunks = text_splitter.split_documents(documents)   # list of smaller chunks (i.e. output of chunking)

    if chunks:
    
        for i, chunk in enumerate(chunks[:5]):           # return first 5 chunks
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks

def create_vector_store(chunks, persist_directory = "db/chroma_db"):
    # Create and persist ChromaDB vector store
    print("Creating embeddings and storing in ChromaDB...")
        
    # embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    print("Main Function")

    #1. Load documents
    documents = load_documents(docs_path="docs")

    #2. Chunk documents
    chunks = split_documents(documents)
    
    #3. Create embeddings and store in vector database
    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()