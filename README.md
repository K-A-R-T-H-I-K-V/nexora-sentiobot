# SentioBot: An Advanced RAG Chatbot for Custom Documentation

<div align="center">
  <img src="[https://api.dicebear.com/7.x/bottts/svg?seed=sentiobot&backgroundColor=00ffff&radius=10](https://api.dicebear.com/7.x/bottts/svg?seed=sentiobot&backgroundColor=00ffff&radius=10)" width="150" alt="SentioBot Logo">
</div>

<p align="center">
  An AI-powered assistant for Nexora Electronics, built with a sophisticated, multi-stage RAG pipeline to provide accurate, context-aware answers from technical manuals and policy documents.
</p>

---
## Table of Contents

* [Live Demo](#live-demo)
* [The Problem](#the-problem-the-limits-of-naive-rag)
* [Our Solution: The Advanced RAG Architecture](#our-solution-the-advanced-rag-architecture)
* [Key Features & Techniques](#key-features--techniques)
* [Tech Stack](#tech-stack)
* [Project Structure](#project-structure)
* [Setup and Installation](#setup-and-installation)
* [How to Run](#how-to-run-the-full-pipeline)
* [Future Improvements & Roadmap](#future-improvements--roadmap)

---
## Live Demo

This is the final, working application, capable of handling specific technical queries and general troubleshooting questions with high accuracy.

![SentioBot in action, successfully answering user questions](https://i.imgur.com/G5g2mNm.png)

---
## The Problem: The Limits of Naive RAG

Simple RAG tutorials often demonstrate a basic "load, split, embed, retrieve" pipeline. While functional for simple keyword matching, this approach quickly fails with complex, real-world documentation. The common failure points we encountered and solved were:

1.  **Fragmented Context:** Arbitrary character-based splitting would break apart tables, lists, and logical sections, leading to incomplete context being sent to the LLM.
2.  **Poor Relevance:** Simple vector search often failed on specific technical queries (e.g., "What is the lifespan?"), prioritizing chunks with more general, high-level prose over dense, factual data.
3.  **Brittle Architecture:** An `InMemoryStore` and random IDs created a system that was inconsistent between runs and slow to start.

---
## Our Solution: The Advanced RAG Architecture

To overcome these challenges, we built a robust, two-stage, persistent architecture that prioritizes semantic meaning and retrieval accuracy.

### Ingestion Pipeline (The "Library Builder")

The data is processed in a multi-step, offline process to prepare it for optimal retrieval.

1.  **Semantic Chunking:** Markdown documents are first split into large "parent" documents based on `##` headings.
2.  **Persistent Parent Storage:** These full-text parent documents are saved to a `LocalFileStore`, providing a persistent, fast-loading "bookshelf" of our full-context documents.
3.  **Batch Summarization:** A stateful `batch_summarize.py` script reads each parent document and uses a powerful LLM (Gemini 1.5 Flash) to generate a dense, keyword-rich summary. This is done in rate-limited batches to stay within API quotas.
4.  **Vector Store Creation:** The final `ingest.py` run takes these high-quality summaries, creates embeddings for them, and stores them in a `ChromaDB` vector store.

### Retrieval Pipeline (The "Librarian")

The Streamlit application is a pure "reader" and is incredibly fast and efficient.

1.  **History-Aware Reformulation:** The user's query and chat history are first passed to an LLM to create a better, standalone question.
2.  **Multi-Query Generation:** The standalone question is then given to an LLM to generate multiple variations from different perspectives (e.g., "What is the lifespan?" becomes "What are the technical specifications?").
3.  **Hybrid Search on Summaries:** All query variations are used in a hybrid search (`BM25` for keywords + `Vector Search` for meaning) against the ChromaDB of **summaries**.
4.  **Parent Document Retrieval:** The system retrieves the best-matching summary, extracts its `doc_id`, and uses it to instantly pull the original, **full-text parent document** from the `LocalFileStore`.
5.  **Answer Generation:** This complete, context-rich parent document is sent to the final LLM to generate the accurate, grounded answer.

---
## Key Features & Techniques

* **Parent-Document Retrieval:** The core "search small, retrieve big" pattern for maximum context.
* **Semantic Chunking:** Splitting documents by Markdown headers (`##`, `###`) instead of arbitrary character counts.
* **Deterministic IDs:** Using `uuid.uuid5` to ensure perfect, consistent linkage between parent and child/summary documents.
* **Persistent Document Stores:** Using `LocalFileStore` and `pickle` to decouple ingestion from application runtime, leading to near-instant app startups.
* **Hybrid Search:** Combining keyword (`BM25`) and semantic (`Vector`) search for robust retrieval across all query types.
* **Multi-Query Retriever:** Automatically generating multiple perspectives of a user's query to drastically improve the chances of finding the correct document.
* **Summarize and Embed:** Creating AI-generated, dense summaries of document sections to serve as a high-quality target for vector search, solving the relevance problem.
* **Batch Processing & Statefulness:** The summarization script is designed to handle API rate limits and can be run multiple times, picking up where it left off.

---
## Tech Stack

* **Framework:** Streamlit
* **LLM Orchestration:** LangChain
* **LLM:** Google Gemini 1.5 Flash
* **Vector Database:** ChromaDB
* **Embedding Model:** HuggingFace `all-MiniLM-L6-v2`
* **Hybrid Search:** `BM25Retriever`
* **Deployment:** Local (easily containerizable with Docker)

---
## Project Structure

```
nexora-sentiobot/
|
├── data/                  # Source documents (.md, .csv)
│   ├── faqs.csv
│   └── ...
├── notebooks/             # Jupyter notebooks for inspection and debugging
│   └── inspect_summaries.ipynb
├── scripts/
│   ├── ingest.py          # Main ingestion script
│   └── batch_summarize.py # Stateful script for generating summaries
├── parent_docstore/       # Persistent storage for full-text parent documents
├── summaries/             # Saved AI-generated summaries
├── vector_db/             # Persistent ChromaDB vector store
├── .env                   # For API keys and environment variables
├── app.py                 # The main Streamlit application
└── requirements.txt       # Python dependencies
```

---
## Setup and Installation

Follow these steps to get the project running on your local machine.

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd nexora-sentiobot
    ```

2.  **Create and Activate Conda Environment**
    ```bash
    conda create --name nexora_env python=3.10
    conda activate nexora_env
    ```

3.  **Install Dependencies**
    First, create a `requirements.txt` file by running this command in your activated environment:
    ```bash
    pip freeze > requirements.txt
    ```
    Then, for any new setup, you can install the dependencies with:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    Create a file named `.env` in the root of the project directory and add your Google API key:
    ```
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```

---
## How to Run: The Full Pipeline

This is a two-stage process. You must run the ingestion pipeline first to build the necessary data stores.

### Stage 1: Initial Ingestion & Summarization

1.  **Run Initial Ingestion**
    This step creates the parent documents needed by the summarizer.
    ```bash
    python scripts/ingest.py
    ```
    *(It is normal for this first run to warn that the vector store is empty.)*

2.  **Generate Summaries in Batches**
    Run this script repeatedly. Each run will process a batch of documents and then pause to respect API limits. Continue until it says "All documents have already been summarized."
    ```bash
    python scripts/batch_summarize.py
    ```

3.  **Run Final Ingestion**
    Once all summaries are created, run the main ingestion script one last time. This will read the summaries and build the final vector store.
    ```bash
    python scripts/ingest.py
    ```

### Stage 2: Launch the Application

After the ingestion process is complete, you can run the Streamlit app.
```bash
streamlit run app.py
```

---
## Future Improvements & Roadmap

This project provides a powerful foundation. Here are some exciting directions to take it next:

### 1. **Citation with Source Highlighting**
* **What:** Instead of just listing the source document, the LLM could be prompted to extract the *exact sentence(s)* from the source that directly support its answer.
* **Why:** This provides "ground truth" and drastically increases user trust. the UI could then highlight this quote within the "View Sources" expander.

### 2. **Multimodal RAG**
* **What:** The current system only processes text. A multimodal system would also ingest images, diagrams, and tables from the documents. Using a multimodal model (like `gemini-pro-vision`), the chatbot could answer questions like, "Show me the wiring diagram for the Thermostat Pro" or "What does the icon for vacation mode look like?"
* **Why:** Many technical manuals rely heavily on visual information. This would unlock a huge portion of currently unused data.

### 3. **Agentic Behavior for Troubleshooting**
* **What:** Convert the chatbot into a simple agent. When a user asks a troubleshooting question like "My light is acting weird," the agent could ask clarifying questions ("Is it flickering, or is it offline in the app?") before retrieving the final, most relevant document.
* **Why:** This creates a more interactive and helpful user experience, guiding the user to the correct solution faster than a single-shot Q&A.

### 4. **Evaluation Pipeline**
* **What:** Implement a RAG evaluation framework like **RAGAs** or **TruLens**. This involves creating a "golden dataset" of questions and ideal answers. The framework can then be used to automatically score the performance of the retrieval and generation steps.
* **Why:** This allows for objective, data-driven improvements. Instead of guessing if a change (like retriever weights) helped, you can measure it scientifically with metrics like context relevance and answer faithfulness.

### 5. **Knowledge Graph Integration**
* **What:** The ultimate upgrade. Instead of storing data as unstructured text chunks, use an LLM to parse all documents into a structured knowledge graph of entities and relationships (e.g., `(LumiGlow Bulb) -[has lifespan of]-> (25,000 hours)`).
* **Why:** This allows for much more complex, multi-hop queries that standard RAG struggles with, such as "Compare the warranty periods and lifespans of all smart light products."