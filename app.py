# app.py - Final Production Version

import os
import re
import pickle
import logging 
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever

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

# --- 2. Model and Retriever Initialization ---
@st.cache_resource(show_spinner="Initializing SentioBot...")
def get_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    db_path = "vector_db"
    store_path = "parent_docstore"
    parent_list_path = "parents.pkl"

    if not all(os.path.exists(p) for p in [db_path, store_path, parent_list_path]):
        st.error("Data stores not found. Please run 'python scripts/ingest.py' first.")
        st.stop()
    
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    byte_store = LocalFileStore(store_path)
    store = create_kv_docstore(byte_store)

    with open(parent_list_path, 'rb') as f:
        all_parent_docs = pickle.load(f)
        
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore, docstore=store,
        id_key="doc_id", child_splitter=child_splitter
    )
    
    bm25_retriever = BM25Retriever.from_documents(all_parent_docs)
    bm25_retriever.k = 10
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[parent_retriever, bm25_retriever], weights=[0.5, 0.5]
    )
    
    llm = get_llm()
    
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever, llm=llm
    )
    
    print("✅ Retriever initialized.")
    return multiquery_retriever

@st.cache_resource(show_spinner=False)
def get_llm():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("Google API key is not set! Please add it to your .env file.")
        st.stop()
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.2)

# --- Formatting Function ---
def format_response(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

# --- 3. Chain Definition ---
def get_context_aware_rag_chain(_retriever):
    llm = get_llm()
    contextualize_q_system_prompt = "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )
    history_aware_retriever = create_history_aware_retriever(llm, _retriever, contextualize_q_prompt)
    
    qa_system_prompt = """## Persona and Role
You are SentioBot, a highly advanced AI customer support assistant for Nexora Electronics. Your persona is professional, precise, and exceptionally helpful.
## Core Directives
1.  **Absolute Grounding:** You MUST base your answers exclusively on the information present in the `Provided Context`. Do not use any external knowledge.
2.  **No Fabrication:** You MUST NOT invent or infer any details, product features, or procedures not explicitly mentioned in the context.
3.  **Fallback Protocol:** If the answer is not in the context, you MUST state clearly: "I do not have enough information to answer that question. For further assistance, please contact our human support team."
## Formatting Rules (Strict)
1.  **Markdown is Mandatory:** You MUST use Markdown for all formatting. Use `##` for main headings and `*` for bullet points. Use `**bold**` for emphasis on key terms.
2.  **Spacing is Critical:** You MUST follow these spacing rules without deviation:
    - Use two newlines (`\n\n`) to separate paragraphs or to separate a heading from a paragraph.
    - Use one newline (`\n`) to separate items in a bulleted list.
    - **NEVER use more than two consecutive newlines.** Your output must be compact and clean.
## Example of Perfect Execution (Few-Shot Example)
---
**User Question:** How do I connect my new thermostat to my Wi-Fi?
**Provided Context:**
Section: Wi-Fi Setup
To connect the Nexora Thermostat Pro, first ensure Bluetooth is enabled on your smartphone. Open the Nexora Home app and tap 'Add Device'. The app will scan for the thermostat. Once found, select it and you will be prompted to enter your Wi-Fi network's password. The device supports 2.4GHz networks only.
**Your Perfect Response:**
I can certainly help you connect your Nexora Thermostat Pro to Wi-Fi.
## Wi-Fi Connection Steps
Please follow these steps using the Nexora Home app on your smartphone:
* **Enable Bluetooth:** Make sure Bluetooth is turned on on your smartphone before you begin.
* **Add Device:** In the app, tap the 'Add Device' button. The app will automatically start scanning for your thermostat.
* **Select and Connect:** Once your thermostat appears in the app, select it and enter your Wi-Fi password when prompted.
**Important Note:** Please ensure you are connecting to a **2.4GHz Wi-Fi network**, as the Thermostat Pro does not support 5GHz networks.
---
Begin!
Context:
{context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# --- 4. Main Application Logic ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! I am SentioBot. How can I assist you with your Nexora devices today?")]
    
retriever = get_retriever()
rag_chain = get_context_aware_rag_chain(retriever)

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

st.title("SentioBot: Your Nexora Electronics Expert")

for msg in st.session_state.chat_history:
    if isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.markdown(msg.content)
            if sources := msg.additional_kwargs.get("sources"):
                with st.expander("View Sources"):
                    for source in sources:
                        st.info(f"Source: {source.metadata.get('source', 'N/A')} | Section: {source.metadata.get('section_title', 'N/A')}")
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(msg.content)

if user_query := st.chat_input("Ask me about Nexora products..."):
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.spinner("Thinking..."):
        try:
            response = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
            raw_answer = response.get("answer", "I do not have enough information to answer.")
            final_answer = format_response(raw_answer)
            sources = response.get("context", [])
            ai_message = AIMessage(content=final_answer, additional_kwargs={"sources": sources})
            st.session_state.chat_history.append(ai_message)
            st.rerun()
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}. Please try again or contact support."
            st.session_state.chat_history.append(AIMessage(content=error_msg))
            st.rerun()
            
            
