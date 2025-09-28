# scripts/ingest.py - Builds data stores using SUMMARIES for retrieval

import os
import shutil
import re
import uuid
import pickle 
from langchain.storage import LocalFileStore 
from langchain.storage._lc_store import create_kv_docstore
from langchain.docstore.document import Document
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Constants ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(PROJECT_ROOT, "vector_db")
PARENT_STORE_PATH = os.path.join(PROJECT_ROOT, "parent_docstore") 
PARENT_LIST_PATH = os.path.join(PROJECT_ROOT, "parents.pkl")
SUMMARIES_PATH = os.path.join(PROJECT_ROOT, "summaries") # Path to the summaries

NAMESPACE_UUID = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8') # This ensures that if we run the script again on the exact same document, it will generate the exact same unique ID, which is very important for consistency.

embedding_model = HuggingFaceEmbeddings( # text-to-numbers
    model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}
)

def process_markdown_semantically(content: str, filename: str) -> list[Document]:
    """
    This function now ONLY processes markdown into PARENT documents.
    Summaries will be used as the children.
    """
    parent_documents = []
    sections = re.split(r'\n(?=## )', content) # It uses a regular expression to split the text every time it sees a newline followed by ##  (which is how Markdown defines a Level 2 heading)
    doc_intro_content = sections[0].strip()
    if doc_intro_content:
        main_title = doc_intro_content.split('\n', 1)[0].replace('# ', '').strip()
        doc_id = str(uuid.uuid5(NAMESPACE_UUID, f"{filename}-{main_title}"))
        parent_doc = Document(
            page_content=doc_intro_content,
            metadata={"source": filename, "section_title": main_title, "doc_id": doc_id}
        )
        parent_documents.append(parent_doc)
    for section_content in sections[1:]:
        if not section_content.strip():
            continue
        lines = section_content.strip().split('\n')
        main_title = lines[0].replace('## ', '').strip()
        doc_id = str(uuid.uuid5(NAMESPACE_UUID, f"{filename}-{main_title}"))
        parent_doc = Document(
            page_content=section_content.strip(),
            metadata={"source": filename, "section_title": main_title, "doc_id": doc_id}
        )
        parent_documents.append(parent_doc)
    return parent_documents

def load_and_create_parents():
    """
    Loads raw files and creates the parent documents.
    """
    all_parents = []
    for root, _, files in os.walk(DATA_PATH):
        for filename in files:
            filepath = os.path.join(root, filename)
            if filename.endswith(".md"):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                parents = process_markdown_semantically(content, filename)
                all_parents.extend(parents)
            elif filename.endswith(".csv"):
                faq_loader = CSVLoader(file_path=filepath, source_column="Question", metadata_columns=["Category"])
                faq_docs = faq_loader.load()
                for doc in faq_docs:
                    question_content = doc.page_content
                    doc_id = str(uuid.uuid5(NAMESPACE_UUID, f"{filename}-{question_content[:50]}"))
                    doc.metadata["doc_id"] = doc_id
                    doc.metadata["source"] = filename
                    all_parents.append(doc)
    return all_parents

def load_summaries_for_retrieval(docstore):
    """
    Loads the generated summaries and creates Document objects for the vector store.
    """
    summary_docs = []
    if not os.path.exists(SUMMARIES_PATH):
        return []
        
    summary_files = os.listdir(SUMMARIES_PATH)
    doc_ids = [f.split('.')[0] for f in summary_files]
    
    # Get the original metadata from the parent docstore
    original_docs = docstore.mget(doc_ids)
    
    for i, doc_id in enumerate(doc_ids):
        summary_file_path = os.path.join(SUMMARIES_PATH, f"{doc_id}.txt")
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            summary_content = f.read()
        
        # The new document for retrieval has the SUMMARY as its content,
        # but the METADATA from the original parent document.
        summary_doc = Document(
            page_content=summary_content,
            metadata=original_docs[i].metadata 
        )
        summary_docs.append(summary_doc)
        
    return summary_docs

def main():
    print("🚀 Starting 2-Stage Ingestion Process...")
    
    # --- Stage 1: Process and save PARENT documents ---
    all_parents = load_and_create_parents()
    
    if os.path.exists(PARENT_STORE_PATH):
        shutil.rmtree(PARENT_STORE_PATH)
    if os.path.exists(PARENT_LIST_PATH):
        os.remove(PARENT_LIST_PATH)
        
    if all_parents:
        print(f"\nCreating persistent parent docstore with {len(all_parents)} documents...")
        byte_store = LocalFileStore(PARENT_STORE_PATH)
        store = create_kv_docstore(byte_store)
        parent_id_map = {doc.metadata["doc_id"]: doc for doc in all_parents}
        store.mset(list(parent_id_map.items()))
        print(f"✅ Parent docstore created successfully.")
        
        print(f"Saving parent document list for BM25 retriever...")
        with open(PARENT_LIST_PATH, 'wb') as f:
            pickle.dump(all_parents, f)
        print(f"✅ Parent list saved to {PARENT_LIST_PATH}.")
    else:
        print("❌ No parent documents found to process.")
        return

    # --- Stage 2: Load summaries and build VECTOR STORE ---
    print("\n---")
    print("Loading summaries to build the retrieval vector store...")
    byte_store = LocalFileStore(PARENT_STORE_PATH)
    docstore = create_kv_docstore(byte_store)
    docs_for_retrieval = load_summaries_for_retrieval(docstore)
    
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        
    if docs_for_retrieval:
        print(f"Found {len(docs_for_retrieval)} summaries to ingest into the vector store.")
        vector_store = Chroma.from_documents(
            documents=docs_for_retrieval, embedding=embedding_model, persist_directory=DB_PATH
        )
        print(f"✅ Vector store created successfully from summaries.")
    else:
        print("⚠️ No summaries found. Vector store is empty. Run `batch_summarize.py` to create summaries.")

if __name__ == "__main__":
    main()
    
