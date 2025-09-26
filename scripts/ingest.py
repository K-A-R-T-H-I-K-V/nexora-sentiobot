# scripts/ingest.py - ETL (Extract, Transform, Load)

import os
import shutil
import re
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Constants (Paths are now robust) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MANUALS_PATH = os.path.join(PROJECT_ROOT, "data", "manuals")
POLICIES_PATH = os.path.join(PROJECT_ROOT, "data", "policies.txt")
FAQS_PATH = os.path.join(PROJECT_ROOT, "data", "faqs.csv")
DB_PATH = os.path.join(PROJECT_ROOT, "vector_db")

# We'll keep the text splitter for non-markdown docs like policies.txt
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}
)

def chunk_markdown_by_section(markdown_text: str, source: str) -> list[Document]:
    """
    A sophisticated chunking function for Markdown files.
    Splits the document by '##' headers, creating a Document for each section
    with rich metadata.
    """
    # Split the document by '##' headers, keeping the header text
    # The regex looks for '\n## ' which is a more reliable split point
    sections = re.split(r'\n## ', markdown_text)
    
    # The first part of the split is the content before the first '##'
    # It might be an introduction or title section
    header_section = sections[0]
    
    # The rest of the splits start with the header title
    remaining_sections = sections[1:]
    
    documents = []
    
    # Handle the initial content (document title, introduction)
    if header_section.strip():
        # The first line is often the main title, let's extract it
        title = header_section.split('\n', 1)[0].replace('# ', '').strip()
        documents.append(Document(
            page_content=header_section.strip(),
            metadata={"source": source, "section_title": title}
        ))
        
    # Process each subsequent section
    for section in remaining_sections:
        if not section.strip():
            continue
        
        # The first line of the section is the title
        parts = section.split('\n', 1)
        title = parts[0].strip()
        content = parts[1].strip() if len(parts) > 1 else ""
        
        # Re-add the header to the content for full context
        full_content = f"## {title}\n{content}"
        
        documents.append(Document(
            page_content=full_content,
            metadata={"source": source, "section_title": title}
        ))
        
    return documents

def load_and_process_documents():
    """Loads all documents and applies the appropriate chunking strategy."""
    all_chunks = []

    # --- Process Markdown Manuals with our custom function ---
    for filename in os.listdir(MANUALS_PATH):
        if filename.endswith(".md"):
            filepath = os.path.join(MANUALS_PATH, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use our sophisticated chunking for Markdown
            md_chunks = chunk_markdown_by_section(content, filename)
            all_chunks.extend(md_chunks)
            print(f"Processed {filename} into {len(md_chunks)} context-aware chunks.")

    # --- Process Policies (using standard text splitter) ---
    policy_loader = TextLoader(POLICIES_PATH)
    policy_docs = policy_loader.load()
    policy_chunks = TEXT_SPLITTER.split_documents(policy_docs)
    all_chunks.extend(policy_chunks)
    print(f"Processed policies into {len(policy_chunks)} chunks.")

    # --- Process FAQs (no chunking needed, they are already atomic) ---
    faq_loader = CSVLoader(file_path=FAQS_PATH, source_column="Question", metadata_columns=["Category"])
    faq_docs = faq_loader.load()
    # You could add the product category to the metadata here if needed
    all_chunks.extend(faq_docs)
    print(f"Loaded {len(faq_docs)} FAQs as individual documents.")
    
    return all_chunks

def main():
    """The main data ingestion pipeline."""
    print("🚀 Starting sophisticated data ingestion process...")
    
    chunks = load_and_process_documents()
    
    # Clean up old database
    if os.path.exists(DB_PATH):
        print(f"Clearing existing database at {DB_PATH}")
        shutil.rmtree(DB_PATH)

    # Create the new vector store
    print(f"Creating vector store with {len(chunks)} total documents...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    print(f"✅ Vector store created successfully with {vector_store._collection.count()} vectors.")

if __name__ == "__main__":
    main()