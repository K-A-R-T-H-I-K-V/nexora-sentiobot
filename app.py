# app.py

import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from streamlit_chat import message

# --- 1. Setup and Configuration ---
load_dotenv()
st.set_page_config(page_title="SentioBot | Nexora Support", page_icon="💡", layout="centered")


# --- UI OVERHAUL: Custom CSS for a professional, modern look ---
def load_css():
    st.markdown("""
        <style>
            /* General body styling */
            .stApp {
                background-color: #0E1117; /* Dark background */
            }
            
            /* Sidebar styling */
            [data-testid="stSidebar"] {
                background-color: #161B22;
                border-right: 1px solid #30363D;
            }
            .st-emotion-cache-16txtl3 { color: #C9D1D9; } /* Sidebar header */
            .st-emotion-cache-1y4p8pa { color: #8B949E; } /* Sidebar info text */

            /* Chat input box styling */
            .stTextInput>div>div>input {
                background-color: #0D1117;
                color: #C9D1D9;
                border: 1px solid #30363D;
                border-radius: 8px;
            }
            .stTextInput>div>div>input:focus {
                border-color: #58A6FF;
                box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.25);
            }

            /* Expander (for sources) styling */
            .stExpander {
                background-color: #161B22;
                border-radius: 8px;
                border: 1px solid #30363D;
            }
            .stExpander header {
                color: #8B949E;
            }
        </style>
    """, unsafe_allow_html=True)

load_css()


# --- 2. Model and Retriever Initialization ---
@st.cache_resource
def get_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="vector_db", embedding_function=embedding_model)
    base_retriever = vector_store.as_retriever(search_kwargs={'k': 10})
    reranker = CohereRerank(model="rerank-english-v3.0", top_n=3)
    compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)
    return compression_retriever

@st.cache_resource
def get_llm():
    """Initializes and returns the Gemini LLM."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("Google API key is not set! Please add it to your .env file.")
        st.stop()
        
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Updated to the current stable, non-premium model (free tier available)
        google_api_key=google_api_key,
        temperature=0.2
    )


# --- 3. Chain Definition (The Core Logic) ---
def get_context_aware_rag_chain(_retriever):
    llm = get_llm()
    contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
    history_aware_retriever = create_history_aware_retriever(llm, _retriever, contextualize_q_prompt)
    qa_system_prompt = """You are SentioBot, an expert customer support assistant for Nexora Electronics. Your tone must be professional, helpful, and friendly. You must answer the user's question based ONLY on the provided context. If the context doesn't contain the answer, you MUST state that you don't have enough information and suggest contacting human support. Never make up information. Be concise and clear in your answers.

Context:
{context}"""
    qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


# --- 4. Main Application Logic ---
retriever = get_retriever()
rag_chain = get_context_aware_rag_chain(retriever)


# --- 5. Beautiful UI and Chat Logic ---
logo_svg = """
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="#58A6FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-zap">
  <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
</svg>
"""
with st.sidebar:
    st.markdown(logo_svg, unsafe_allow_html=True)
    st.header("About SentioBot")
    st.info("An AI-powered assistant for Nexora Electronics, providing instant support from official documentation.")
    st.header("Tech Stack")
    st.markdown("- Streamlit\n- LangChain\n- Google Gemini\n- ChromaDB\n- Cohere")

st.title("SentioBot: Your Nexora Electronics Expert")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! I am SentioBot. How can I assist you with your Nexora devices today?")]

# --- FINAL UI FIX: Reverted to the standard, reliable chat input method ---
if user_query := st.chat_input("Ask me about Nexora products..."):
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.spinner("Thinking..."):
        try:
            response = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
            ai_response_content = response["answer"]
            sources = response["context"]
            ai_message = AIMessage(content=ai_response_content, additional_kwargs={"sources": sources})
            st.session_state.chat_history.append(ai_message)
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}. Please try again or contact support."
            st.session_state.chat_history.append(AIMessage(content=error_msg))
            st.error(error_msg)

# Display the entire chat history after processing the new message
for i, msg in enumerate(st.session_state.chat_history):
    if isinstance(msg, HumanMessage):
        message(msg.content, is_user=True, key=f"user_msg_{i}", avatar_style="adventurer-neutral")
    else: # AIMessage
        message(msg.content, is_user=False, key=f"ai_msg_{i}", avatar_style="bottts")
        if msg.additional_kwargs.get("sources"):
            with st.expander("View Sources"):
                for source in msg.additional_kwargs["sources"]:
                    st.info(f"Source: {source.metadata.get('source', 'N/A')} | Section: {source.metadata.get('section_title', 'N/A')}")

