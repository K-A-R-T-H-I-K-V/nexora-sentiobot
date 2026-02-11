import os
import shutil
import re
import uuid
import pickle
from typing import List, Any
from langchain.storage import LocalFileStore 
from langchain.storage._lc_store import create_kv_docstore
from langchain.docstore.document import Document
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# config 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(PROJECT_ROOT, "vector_db")
PARENT_STORE_PATH = os.path.join(PROJECT_ROOT, "parent_docstore") 
PARENT_LIST_PATH = os.path.join(PROJECT_ROOT, "parents.pkl")
SUMMARIES_PATH = os.path.join(PROJECT_ROOT, "summaries")

# Static namespace for generating deterministic UUIDs.
# This ensures idempotency: re-ingesting the same file yields the exact same IDs, preventing duplicates.
NAMESPACE_UUID = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}
)

def process_markdown_semantically(content: str, filename: str) -> List[Document]:
    """
    Parses raw Markdown into 'Parent' documents based on header structure.
    
    Note: These documents are NOT vectorised directly. They serve as the 
    ground truth context. We will later vectorise their *summaries* to 
    decouple retrieval from context window usage.
    """
    parent_documents: List[Document] = [] 
    
    # Split text by Level 2 headers (##) to preserve logical section boundaries.
    # We use a lookahead assertion to keep the delimiter.
    sections = re.split(r'\n(?=## )', content)
    
    # Process the introduction (content before the first H2)
    doc_intro_content = sections[0].strip()
    if doc_intro_content:
        # Extract title from the first line for metadata
        main_title = doc_intro_content.split('\n', 1)[0].replace('# ', '').strip()
        
        # Generate a deterministic ID based on content source
        doc_id = str(uuid.uuid5(NAMESPACE_UUID, f"{filename}-{main_title}"))
        
        parent_doc = Document(
            page_content=doc_intro_content,
            metadata={
                "source": filename, 
                "section_title": main_title, 
                "doc_id": doc_id
            }
        )
        parent_documents.append(parent_doc)
        
    # Process subsequent sections
    for section_content in sections[1:]:
        if not section_content.strip():
            continue
            
        lines = section_content.strip().split('\n')
        main_title = lines[0].replace('## ', '').strip()
        doc_id = str(uuid.uuid5(NAMESPACE_UUID, f"{filename}-{main_title}"))
        
        parent_doc = Document(
            page_content=section_content.strip(),
            metadata={
                "source": filename, 
                "section_title": main_title, 
                "doc_id": doc_id
            }
        )
        parent_documents.append(parent_doc)
        
    return parent_documents

def load_and_create_parents() -> List[Document]:
    """
    Traverses the data directory to load source files (Markdown/CSV)
    and constructs the primary Parent Document objects.
    """
    all_parents: List[Document] = [] 
    
    for root, _, files in os.walk(DATA_PATH):
        for filename in files:
            filepath = os.path.join(root, filename)
            
            if filename.endswith(".md"):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                parents = process_markdown_semantically(content, filename)
                all_parents.extend(parents)
                
            elif filename.endswith(".csv"):
                # For CSVs, we treat each row as a distinct parent document
                faq_loader = CSVLoader(
                    file_path=filepath, 
                    source_column="Question", 
                    metadata_columns=["Category"]
                )
                faq_docs = faq_loader.load()
                
                for doc in faq_docs:
                    question_content = doc.page_content
                    # Use a slice of the content to ensure uniqueness in ID generation
                    doc_id = str(uuid.uuid5(NAMESPACE_UUID, f"{filename}-{question_content[:50]}"))
                    
                    doc.metadata["doc_id"] = doc_id
                    doc.metadata["source"] = filename
                    all_parents.append(doc)
                    
    return all_parents

def load_summaries_for_retrieval(docstore: Any) -> List[Document]:
    """
    Rehydrates the Retrieval Documents by combining:
    1. Content: The generated AI summary (from disk).
    2. Metadata: The original source metadata (from the Parent Docstore).
    
    This linkage ensures that when we retrieve a summary, we can still
    trace it back to the exact source file and section.
    """
    summary_docs: List[Document] = []
    
    if not os.path.exists(SUMMARIES_PATH):
        return []
        
    summary_files = os.listdir(SUMMARIES_PATH)
    
    # Filenames map directly to doc_ids
    doc_ids = [f.split('.')[0] for f in summary_files]
    
    # Batch retrieve original parent docs to get valid metadata
    original_docs = docstore.mget(doc_ids)
    
    for i, doc_id in enumerate(doc_ids):
        summary_file_path = os.path.join(SUMMARIES_PATH, f"{doc_id}.txt")
        
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_content = f.read()
        
        # Only proceed if the parent exists (data integrity check)
        if original_docs[i]: 
            summary_doc = Document(
                page_content=summary_content,
                # Critical: Inject original metadata so the RAG pipeline knows the source
                metadata=original_docs[i].metadata 
            )
            summary_docs.append(summary_doc)
        
    return summary_docs

def main() -> None:
    print("Starting 2-Stage Ingestion Pipeline...")
    
    #Stage 1: Build the Parent Document Store (Source of Truth)
    all_parents = load_and_create_parents()
    
    # Clean up existing stores to avoid stale data
    if os.path.exists(PARENT_STORE_PATH):
        shutil.rmtree(PARENT_STORE_PATH)
    if os.path.exists(PARENT_LIST_PATH):
        os.remove(PARENT_LIST_PATH)
        
    if all_parents:
        print(f"Indexing {len(all_parents)} parent documents...")
        
        # Persist parents to a local Key-Value store
        byte_store = LocalFileStore(PARENT_STORE_PATH)
        store = create_kv_docstore(byte_store)
        parent_id_map = {doc.metadata["doc_id"]: doc for doc in all_parents}
        store.mset(list(parent_id_map.items()))
        print("Parent KV store built successfully.")
        
        # Serialize list for potential hybrid search (BM25) usage later
        with open(PARENT_LIST_PATH, 'wb') as f:
            pickle.dump(all_parents, f)
        print(f"Parent document list serialized to {PARENT_LIST_PATH}.")
    else:
        print("No parent documents found. Check the /data directory.")
        return

    #Stage 2: Build the Vector Store using Summaries 
    print("\nBuilding Vector Index from Summaries...")
    
    byte_store = LocalFileStore(PARENT_STORE_PATH)
    docstore = create_kv_docstore(byte_store)
    
    docs_for_retrieval = load_summaries_for_retrieval(docstore)
    
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        
    if docs_for_retrieval:
        print(f"Ingesting {len(docs_for_retrieval)} summaries into ChromaDB...")
        
        vector_store = Chroma.from_documents(
            documents=docs_for_retrieval, 
            embedding=embedding_model, 
            persist_directory=DB_PATH
        )
        print("Vector store creation complete.")
    else:
        print("No summaries found. Please run the summarization pipeline (`batch_summarize.py`) before indexing.")

if __name__ == "__main__":
    main()