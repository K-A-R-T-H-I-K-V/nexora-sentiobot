# SentioBot: An Advanced RAG Chatbot for Custom Documentation

<div align="center">
  <img src="[https://api.dicebear.com/7.x/bottts/svg?seed=sentiobot&backgroundColor=00ffff&radius=10](https://api.dicebear.com/7.x/bottts/svg?seed=sentiobot&backgroundColor=00ffff&radius=10)" width="150" alt="SentioBot Logo">
</div>

<p align="center">
  An AI-powered assistant for Nexora Electronics, built with a sophisticated, multi-stage RAG pipeline to provide accurate, context-aware answers from technical manuals and policy documents.
</p>

---
## Table of Contents

  * [Live Demo](https://www.google.com/search?q=%23live-demo)
  * [The Problem](https://www.google.com/search?q=%23the-problem)
      * [Part 1: The Limits of Naive RAG](https://www.google.com/search?q=%23part-1-the-limits-of-naive-rag)
      * [Part 2: The Passive Assistant](https://www.google.com/search?q=%23part-2-the-passive-assistant)
  * [Our Solution: A Multi-Layered Architecture](https://www.google.com/search?q=%23our-solution-a-multi-layered-architecture)
      * [Layer 1: The Advanced RAG Foundation](https://www.google.com/search?q=%23layer-1-the-advanced-rag-foundation)
      * [Layer 2: The Proactive Agent Framework](https://www.google.com/search?q=%23layer-2-the-proactive-agent-framework)
  * [Key Features & Techniques](https://www.google.com/search?q=%23key-features--techniques)
      * [Retrieval (RAG) Features](https://www.google.com/search?q=%23retrieval-rag-features)
      * [Agent & UX Features](https://www.google.com/search?q=%23agent--ux-features)
  * [Tech Stack](https://www.google.com/search?q=%23tech-stack)
  * [Project Structure](https://www.google.com/search?q=%23project-structure)
  * [Setup and Installation](https://www.google.com/search?q=%23setup-and-installation)
  * [How to Run](https://www.google.com/search?q=%23how-to-run)
      * [Stage 1: Ingestion Pipeline](https://www.google.com/search?q=%23stage-1-ingestion-pipeline)
      * [Stage 2: Run the System](https://www.google.com/search?q=%23stage-2-run-the-system)
  * [Future Improvements & Roadmap](https://www.google.com/search?q=%23future-improvements--roadmap)
  
---
## Live Demo

This is the final, working application, capable of handling specific technical queries and general troubleshooting questions with high accuracy.

![SentioBot in action, successfully answering user questions](https://i.imgur.com/G5g2mNm.png)

---
 ---

## The Problem

### Part 1: The Limits of Naive RAG

Simple RAG pipelines fail with complex, real-world documentation. The common failure points we solved were:

1.  **Fragmented Context:** Arbitrary splitting breaks apart tables and logical sections, leading to incomplete answers.
2.  **Poor Relevance:** Simple vector search often fails on specific technical queries, prioritizing general prose over dense, factual data.
3.  **Stateless Inefficiency:** In-memory stores cause slow "cold starts" and inconsistent behavior between runs.

### Part 2: The Passive Assistant

Even a perfect RAG system is just a librarian—it can find the right book, but it can't act on the information for you. This creates a frustrating user experience:

  - **User:** "Is my product under warranty?"
  - **Passive Bot:** "Our warranty policy is for two years." (Unhelpful)
  - **User:** "Okay, can you start a return for me?"
  - **Passive Bot:** "Our return policy states you must contact support." (Frustrating)

The bot could inform, but it couldn't **do**. It lacked agency, memory, and personalization.

-----

## Our Solution: A Multi-Layered Architecture

We built a robust, two-layer system. A powerful RAG pipeline serves as the foundational knowledge layer, while a LangChain Agent acts as the intelligent reasoning and action layer on top.

### Layer 1: The Advanced RAG Foundation (The "Library")

Our ingestion pipeline processes documents for optimal retrieval, forming the agent's long-term memory.

1.  **Semantic Chunking:** Documents are split into "parent" documents by Markdown headings.
2.  **Persistent Parent Storage:** Full-text parent documents are saved to a `LocalFileStore`.
3.  **Vector Store Creation:** We use `HuggingFaceEmbeddings` and store them in a persistent `ChromaDB` vector store.
4.  **Hybrid Search Index:** A `BM25Retriever` index is built for keyword-based search.

### Layer 2: The Proactive Agent Framework (The "Concierge")

The Streamlit app runs a stateful, reasoning agent that uses a suite of tools to solve problems.

1.  **The "Agent's Mind" (Prompt Engineering):** A meticulously crafted prompt acts as the agent's constitution, defining its persona, rules of engagement, and proactive nature.
2.  **Tool Kit:** The RAG pipeline is demoted to be just one tool (`lookup_documentation`). Other tools allow the agent to act:
      - `check_order_status(order_id)`
      - `check_warranty_status(serial_number)`
      - `create_support_ticket(summary)`
3.  **Conversational Memory:** The agent uses `ConversationBufferWindowMemory` to maintain context across multiple turns, enabling coherent, multi-step problem-solving.
4.  **Personalized Context:** On login, the user's profile (including specific products they own and their serial numbers) is injected into the agent's context, allowing for hyper-personalized, proactive assistance.

-----

## Key Features & Techniques

### Retrieval (RAG) Features

  - **Parent-Document Retrieval:** The "search small, retrieve big" pattern. We perform a hybrid search on child chunks to find and return the full-context parent document.
  - **Hybrid Search:** Combining keyword (`BM25`) and semantic (`Vector`) search for robust retrieval across all query types.
  - **Multi-Query Retriever:** Automatically generating multiple perspectives of a user's query to drastically improve the chances of finding the correct document.
  - **Persistent Document Stores:** Using `LocalFileStore` and `ChromaDB` to decouple ingestion from runtime, leading to near-instant app startups.

### Agent & UX Features

  - **Tool-Using Agent (LangChain Agents):** The agent can reason, plan, and use a suite of tools to execute tasks like checking a warranty or creating a support ticket.
  - **Conversational Memory:** The agent remembers previous turns in the conversation, eliminating frustrating loops and allowing it to handle complex, multi-step user requests.
  - **User Personalization & Proactivity:** A login system provides the agent with the user's profile. The agent is explicitly instructed to use this data (e.g., product serial numbers) proactively to save the user time.
  - **Feedback Loop & Analytics:** Interactive 👍/👎 buttons on each response log user feedback to `analytics.log`. A separate `dashboard.py` visualizes this data, providing insights into user pain points and knowledge gaps.
  - **Robust Error Handling:** The `AgentExecutor` is configured with a self-correction mechanism, allowing it to recover from intermittent LLM formatting errors, making the system significantly more reliable.

-----

## Tech Stack

  - **Framework:** Streamlit
  - **LLM Orchestration:** LangChain
  - **LLM:** Google Gemini 1.5 Flash
  - **Vector Database:** ChromaDB
  - **Embedding Model:** HuggingFace `all-MiniLM-L6-v2`
  - **Hybrid Search:** `BM25Retriever`
  - **Analytics:** Pandas

## Project Structure

```
nexora-sentiobot/
|
├── data/                  # Source documents (.md, .csv)
├── scripts/
│   ├── ingest.py          # Main script to build the document stores and vector DB
│   └── ...
├── parent_docstore/       # Persistent storage for full-text parent documents
├── vector_db/             # Persistent ChromaDB vector store
|
├── .env                   # For API keys and environment variables
├── app.py                 # The main Streamlit application (the agent)
├── dashboard.py           # The Streamlit analytics dashboard
├── tools.py               # Defines the tools the agent can use
├── mock_db.py             # A mock database for users, products, and orders
|
├── analytics.log          # Log file for user interactions and feedback
├── support_tickets.log    # Log file for created support tickets
└── requirements.txt       # Python dependencies
```

-----
## Setup and Installation

Follow these steps to get the project running on your local machine.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/K-A-R-T-H-I-K-V/nexora-sentiobot.git
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
    Create a file named `.env` in the root of the project directory and add your Google API key, as well as your cohere key:
    ```
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    COHERE_API_KEY="YOUR_COHERE_API_KEY_HERE"
    ```

---
## How to Run: The Full Pipeline

This is a multi-stage process. You must run the ingestion pipeline first to build the necessary data stores.

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

### Stage 2: Run the System

The application and the dashboard run in separate processes.

1.  **Launch the Chatbot Application**
    Open a terminal, activate your environment, and run:

    ```bash
    streamlit run app.py
    ```

2.  **Launch the Analytics Dashboard**
    Open a **second terminal**, activate the environment, and run:

    ```bash
    streamlit run dashboard.py
    ```

---
## Future Improvements & Roadmap

This project provides a powerful foundation. Here are some exciting directions to take it next:

### 1. **Citation with Source Highlighting**
* **What:** Instead of just listing the source document, the LLM could be prompted to extract the *exact sentence(s)* from the source that directly support its claim and highlight it in the UI to build user trust.
* **Why:** This provides "ground truth" and drastically increases user trust. the UI could then highlight this quote within the "View Sources" expander.

### 2. **Multimodal RAG**
* **What:** The current system only processes text. A multimodal system would also ingest images, diagrams, and tables from the documents. Using a multimodal model (like `gemini-pro-vision`), the chatbot could answer questions like, "Show me the wiring diagram for the Thermostat Pro" or "What does the icon for vacation mode look like?"
* **Why:** Many technical manuals rely heavily on visual information. This would unlock a huge portion of currently unused data.


### 3. **Evaluation Pipeline**
* **What:** Implement a RAG evaluation framework like **RAGAs** or **TruLens**. This involves creating a "golden dataset" of questions and ideal answers. The framework can then be used to automatically score the performance of the retrieval and generation steps.
* **Why:** This allows for objective, data-driven improvements. Instead of guessing if a change (like retriever weights) helped, you can measure it scientifically with metrics like context relevance and answer faithfulness.

### 4. **Knowledge Graph Integration**
* **What:** The ultimate upgrade. Instead of storing data as unstructured text chunks, use an LLM to parse all documents into a structured knowledge graph of entities and relationships (e.g., `(LumiGlow Bulb) -[has lifespan of]-> (25,000 hours)`).
* **Why:** This allows for much more complex, multi-hop queries that standard RAG struggles with, such as "Compare the warranty periods and lifespans of all smart light products."

Of course. Your project has evolved significantly from an advanced RAG pipeline into a full-fledged, personalized AI agent. The README.md should reflect this incredible progress.

I have completely rewritten and expanded your README to be highly detailed and informative, showcasing the full scope of what SentioBot can do. It now tells the complete story, from the foundational RAG architecture to the intelligent agent layer you built on top.

-----




