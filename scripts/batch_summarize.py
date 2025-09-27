# scripts/batch_summarize.py (True Batching with Structured Output)

import os
import time
import math
from dotenv import load_dotenv
from typing import List # Required for list type hinting
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# --- Configuration ---
BATCH_SIZE = 10 # How many documents to process in a single API call
DELAY_BETWEEN_BATCHES = 65 # Seconds to wait to respect API rate limits
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PARENT_STORE_PATH = os.path.join(PROJECT_ROOT, "parent_docstore")
SUMMARIES_PATH = os.path.join(PROJECT_ROOT, "summaries")

# --- Setup ---
load_dotenv()
os.makedirs(SUMMARIES_PATH, exist_ok=True)

# <<< START: Pydantic schema for structured output >>>
# This defines the structure we want the LLM to return.
class Summary(BaseModel):
    """A single, dense, keyword-rich summary of a document section."""
    summary_text: str = Field(description="The generated summary, optimized for search.")

class Summaries(BaseModel):
    """A list of summaries, corresponding to a list of input documents."""
    summaries: List[Summary]
# <<< END: Pydantic schema >>>

# --- LLM and Prompt for Summarization ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# This new prompt is designed to instruct the LLM on handling multiple docs
multi_doc_prompt = ChatPromptTemplate.from_template(
    "You are an expert at creating concise, information-dense summaries of technical documents for a Retrieval Augmented Generation (RAG) system."
    "\n\nYour task is to summarize each of the following {num_docs} document sections, provided below. "
    "For each document, create a summary that includes all key entities, product names, technical specifications, and important keywords."
    "The summary must be optimized for semantic vector search and should not contain any conversational fluff."
    "\n\nRespond with a list of summaries, one for each document in the order they were provided."
    "\n\nHere are the documents:\n\n{documents}"
)

# We bind the LLM to our Pydantic schema to force structured output
structured_llm = llm.with_structured_output(Summaries)
summarize_chain = multi_doc_prompt | structured_llm

def main():
    print("🚀 Starting AUTOMATED batch summarization with GUARANTEED structured output...")
    
    byte_store = LocalFileStore(PARENT_STORE_PATH)
    docstore = create_kv_docstore(byte_store)
    
    existing_summary_ids = [f.split('.')[0] for f in os.listdir(SUMMARIES_PATH)]
    all_parent_ids = list(byte_store.yield_keys())
    docs_to_summarize_ids = [doc_id for doc_id in all_parent_ids if doc_id not in existing_summary_ids]
    
    if not docs_to_summarize_ids:
        print("✅ All documents have already been summarized.")
        return
        
    print(f"Found {len(docs_to_summarize_ids)} documents needing summarization.")
    
    batch_num = 1
    total_batches = math.ceil(len(docs_to_summarize_ids) / BATCH_SIZE)
    
    while docs_to_summarize_ids:
        print("\n" + "="*50)
        print(f"Processing Batch {batch_num} of {total_batches}...")
        
        batch_ids_to_process = docs_to_summarize_ids[:BATCH_SIZE]
        batch_docs = docstore.mget(batch_ids_to_process)
        
        # <<< START: New logic for single API call per batch >>>
        # Combine all documents in the batch into a single string
        formatted_docs_string = ""
        for i, doc in enumerate(batch_docs):
            formatted_docs_string += f"--- DOCUMENT {i+1} ---\n{doc.page_content}\n\n"
        
        try:
            print(f"  - Sending {len(batch_docs)} documents to the API in a single request...")
            # Make ONE API call for the entire batch
            result = summarize_chain.invoke({
                "num_docs": len(batch_docs),
                "documents": formatted_docs_string
            })
            
            # The result is a Pydantic object, guaranteed to have the right structure
            output_summaries = result.summaries
            
            if len(output_summaries) != len(batch_docs):
                print(f"  ❌ Mismatch Error: Got {len(output_summaries)} summaries for {len(batch_docs)} documents. Skipping batch.")
                continue

            # Loop through the guaranteed results and save each summary
            for i, summary_obj in enumerate(output_summaries):
                doc_id = batch_docs[i].metadata["doc_id"]
                summary_text = summary_obj.summary_text
                
                summary_filepath = os.path.join(SUMMARIES_PATH, f"{doc_id}.txt")
                with open(summary_filepath, 'w', encoding='utf-8') as f:
                    f.write(summary_text)
            
            print(f"✅ Batch {batch_num} complete. Generated and saved {len(output_summaries)} new summaries.")

        except Exception as e:
            print(f"  ❌ An error occurred during batch API call: {e}")
        # <<< END: New logic >>>
        
        docs_to_summarize_ids = docs_to_summarize_ids[len(batch_ids_to_process):]
        batch_num += 1
        
        if docs_to_summarize_ids:
            print(f"--- WAITING {DELAY_BETWEEN_BATCHES} seconds before next batch ---")
            time.sleep(DELAY_BETWEEN_BATCHES)
            
    print("\n" + "="*50)
    print("🎉 ALL DOCUMENTS HAVE BEEN SUMMARIZED. PROCESS COMPLETE. 🎉")
    
if __name__ == "__main__":
    main()