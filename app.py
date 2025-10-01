import os
import re
import pickle
import logging
import json
from datetime import datetime, timedelta
from typing import List
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.retrievers.multi_query import MultiQueryRetriever
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain.docstore.document import Document
from langchain.tools import tool 
from langchain.agents import AgentExecutor, create_react_agent
from tools import check_order_status, check_warranty_status, create_support_ticket
from langchain.retrievers import EnsembleRetriever
from langchain import hub

# Set up logging to see the generated queries in the terminal (optional but helpful)
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# --- 1. Setup and Configuration ---
load_dotenv()
st.set_page_config(page_title="SentioBot | Nexora Support", page_icon="💡", layout="centered")

# --- Avatar URLs and CSS ---
USER_AVATAR = "https://api.dicebear.com/7.x/adventurer/svg?seed=user"
ASSISTANT_AVATAR = "https://api.dicebear.com/7.x/bottts/svg?seed=sentiobot&backgroundColor=00ffff"
def load_css():
    st.markdown("""
        <style>
            .stApp { background-color: #0E1117; }
            [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
            .stTextInput>div>div>input { background-color: #0D1117; border: 1px solid #30363D; border-radius: 8px; }
            .stExpander { background-color: #161B22; border-radius: 8px; border: 1px solid #30363D; }
            .stChatMessage { animation: fadeIn 0.5s; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        </style>
    """, unsafe_allow_html=True)
load_css()

# --- Pydantic schemas for structured citation output ---
class Citation(BaseModel):
    source_id: int = Field(description="The integer index of the source document that supports the claim, starting from 1.")
    claim: str = Field(description="The specific claim or statement from the answer that is directly supported by this source.")

class AnswerWithCitations(BaseModel):
    answer: str = Field(description="The final answer to the user's question, written in Markdown.")
    citations: List[Citation] = Field(description="A list of all claims and their corresponding source document IDs.")

# --- 2. Model and Retriever Initialization ---
@st.cache_resource(show_spinner="Initializing SentioBot...")
def get_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db_path, store_path, parent_list_path = "vector_db", "parent_docstore", "parents.pkl"
    if not all(os.path.exists(p) for p in [db_path, store_path, parent_list_path]):
        st.error("Data stores not found. Please run 'python scripts/ingest.py' first.")
        st.stop()
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    byte_store = LocalFileStore(store_path)
    store = create_kv_docstore(byte_store)
    with open(parent_list_path, 'rb') as f:
        all_parent_docs = pickle.load(f)

    # Note: We are not using ParentDocumentRetriever directly in the agent,
    # but keeping the setup here in case you want to switch back or test.
    # The multiquery retriever is what we will pass to the agent.
    bm25_retriever = BM25Retriever.from_documents(all_parent_docs)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )

    template = """You are an AI language model assistant. Your task is to generate 3
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of distance-based similarity search.
    Provide these alternative questions separated by newlines.
    Original question: {question}"""
    prompt_perspectives = PromptTemplate.from_template(template)

    llm = get_llm()
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever, 
        llm=llm,
        prompt=prompt_perspectives
    )
    print("✅ Retriever initialized.")
    return multiquery_retriever

@st.cache_resource(show_spinner=False)
def get_llm():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("Google API key is not set! Please add it to your .env file.")
        st.stop()
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.1)

# --- Formatting Functions ---
def format_answer_with_citations(response_obj: AnswerWithCitations, sources: list) -> str:
    formatted_answer = response_obj.answer
    cited_source_ids = sorted(list(set(c.source_id for c in response_obj.citations)))
    citation_markers = "".join(f" [{i+1}]" for i in range(len(cited_source_ids)))
    if citation_markers:
        formatted_answer += f" {citation_markers}"

    citation_references = []
    for i, source_id in enumerate(cited_source_ids):
        if 1 <= source_id <= len(sources):
            source_doc = sources[source_id - 1]
            source_name = source_doc.metadata.get('source', 'N/A')
            section_name = source_doc.metadata.get('section_title', 'N/A')
            citation_references.append(f"[{i+1}] {source_name} | Section: {section_name}")

    if citation_references:
        formatted_answer += "\n\n---\n**Sources:**\n" + "\n".join(citation_references)
    return formatted_answer

def format_docs_with_ids(docs: List[Document]) -> str:
    return "\n\n".join(f"---\nSource ID: {i+1}\nContent: {doc.page_content}\n---" for i, doc in enumerate(docs))

# --- 3. Agent and Tools Definition ---
@st.cache_resource(show_spinner="Initializing Agent...")
def get_agent_executor(_retriever, chat_history: List):
    """Creates the Agent with all its tools, including the RAG chain."""
    llm = get_llm()

    @tool
    def lookup_documentation(query: str) -> str:
        """
        Use this tool to answer general questions about Nexora products, policies,
        troubleshooting guides, and technical specifications. It searches the
        official documentation. Use this for any question that does not involve
        a specific order ID, serial number, or a request for a human.
        """

        if not query or query.strip() == "":
            return "I cannot look up documentation without a specific question. Please provide more details."

        # New detailed log for the RAG tool
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "="*60)
        print(f"[{timestamp}] Executing Tool: lookup_documentation")
        print("-"*60)
        print(f"INPUT QUERY: {query}")

        structured_llm_rag = llm.with_structured_output(AnswerWithCitations)
        
        qa_system_prompt = """## Persona and Role
You are SentioBot, a highly advanced AI customer support assistant for Nexora Electronics. Your persona is professional, precise, and exceptionally helpful.
## Core Directives
1.  **Analyze the `Provided Context`:** The context contains several source documents. Each document is clearly marked with a `Source ID` (e.g., `Source ID: 1`).
2.  **Generate a Comprehensive Answer:** Synthesize the information from the provided documents to answer the user's `Question`.
3.  **Cite Your Sources:** For every factual claim or specific piece of information in your answer, you MUST identify which `Source ID` it came from.
4.  **Format the Output:** You MUST format your entire output as a single, valid JSON object that strictly follows the provided `AnswerWithCitations` schema. Do not add any text or formatting outside of this JSON object.
5.  **Use Markdown:** The text within the `answer` field of your JSON output MUST be formatted using Markdown. Use `##` for headings, `*` for bullet points, and `**bold**` for emphasis on key terms.

## `Provided Context` Example
---
Source ID: 1
Content: The Nexora Thermostat Pro has a 2-year warranty.
---
---
Source ID: 2
Content: To reset the LumiGlow bulb, turn it on and off 5 times.
---
## `Question` Example
"What is the warranty on the thermostat and how do I reset the light?"
## Perfect Output Example (Must be a single JSON object)
{{
  "answer": "The Nexora Thermostat Pro comes with a **2-year warranty**. To reset the LumiGlow bulb, you need to turn it on and off five times in a row.",
  "citations": [
    {{
      "source_id": 1,
      "claim": "The Nexora Thermostat Pro comes with a 2-year warranty."
    }},
    {{
      "source_id": 2,
      "claim": "To reset the LumiGlow bulb, you need to turn it on and off five times in a row."
    }}
  ]
}}
---
Begin!

`Provided Context`:
{context}

`Question`:
{input}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [("system", qa_system_prompt), MessagesPlaceholder(variable_name="chat_history")]
        )
        docs = _retriever.invoke(query)

        # New detailed log of retrieved documents
        retrieved_sources = [f"{doc.metadata.get('source', 'N/A')} | Section: {doc.metadata.get('section_title', 'N/A')}" for doc in docs]
        print("\nRETRIEVED CONTEXT:")
        print(json.dumps(retrieved_sources, indent=2))
        print("="*60 + "\n")

        rag_chain = (
            RunnablePassthrough.assign(context=(lambda x: format_docs_with_ids(x["context"])))
            | qa_prompt
            | structured_llm_rag
        )
        response = rag_chain.invoke({
            "context": docs, 
            "input": query, 
            "chat_history": chat_history
        })
        if isinstance(response, AnswerWithCitations):
            return format_answer_with_citations(response, docs)
        return "I found some information in the documentation, but couldn't structure it correctly."

    tools = [
        lookup_documentation,
        check_order_status,
        check_warranty_status,
        create_support_ticket
    ]
    
    # FIX 2: Pull the official, compatible prompt from LangChain Hub
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm, tools, prompt) # <-- FIX 3: Use the new prompt here
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 4. Main Application Logic ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! I am SentioBot. How can I assist you with your Nexora devices today?")]

retriever = get_retriever()
agent_executor = get_agent_executor(retriever, st.session_state.chat_history)

# --- 5. UI and Chat Logic ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center;">
        <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24" fill="none" stroke="#58A6FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
        </svg>
    </div>
    """, unsafe_allow_html=True)
    st.header("About SentioBot")
    st.info("An AI-powered assistant for Nexora Electronics, providing instant support from official documentation.")
    st.header("Tech Stack")
    st.markdown("- Streamlit\n- LangChain\n- Google Gemini\n- ChromaDB\n- BM25")
    
    # NEW FEATURE: Clear Chat History Button
    if st.button("Clear Conversation"):
        st.session_state.chat_history = [AIMessage(content="Hello! I am SentioBot. How can I assist you with your Nexora devices today?")]
        st.rerun()

st.title("SentioBot: Your Nexora Electronics Expert")

for msg in st.session_state.chat_history:
    avatar = USER_AVATAR if isinstance(msg, HumanMessage) else ASSISTANT_AVATAR
    with st.chat_message(msg.type, avatar=avatar):
        st.markdown(msg.content)

if user_query := st.chat_input("Ask me about Nexora products..."):
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Thinking..."):
            try:
                response = agent_executor.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                final_answer = response.get("output", "I'm sorry, I encountered an error.")
                st.markdown(final_answer)
                ai_message = AIMessage(content=final_answer)
                st.session_state.chat_history.append(ai_message)
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}. Please try again."
                st.error(error_msg)
                st.session_state.chat_history.append(AIMessage(content=error_msg))
