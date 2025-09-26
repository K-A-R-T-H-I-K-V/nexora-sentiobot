# app.py

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# --- NEW IMPORTS FOR RE-RANKING ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# --- 1. Setup and Configuration ---
load_dotenv()
st.set_page_config(page_title="SentioBot for Nexora", page_icon="🤖")
st.title("🤖 SentioBot: Your Nexora Electronics Expert")
st.caption("I'm powered by Google Gemini and trained on Nexora's official documentation.")

# --- 2. Model and Retriever Initialization (with Re-ranking Upgrade) ---
@st.cache_resource
def get_retriever():
    """Initializes and returns the Chroma vector store retriever with re-ranking."""
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="vector_db", embedding_function=embedding_model)

    # a. Base Retriever: Fetches a larger pool of documents initially.
    base_retriever = vector_store.as_retriever(search_kwargs={'k': 10})

    # b. Re-ranker: Initializes the Cohere Re-rank model.
    reranker = CohereRerank(top_n=3)

    # c. Compression Retriever: The final, enhanced retriever.
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base_retriever
    )

    return compression_retriever

@st.cache_resource
def get_llm():
    """Initializes and returns the Gemini Flash LLM."""
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1, convert_system_message_to_human=True)

# --- 3. Chain Definition ---
def get_context_aware_rag_chain(_retriever):
    """Creates and returns the full conversational RAG chain."""
    llm = get_llm()

    contextualize_q_system_prompt = """
    Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history.
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, _retriever, contextualize_q_prompt)

    qa_system_prompt = """
    You are SentioBot, an expert customer support assistant for Nexora Electronics.
    Your tone must be professional, helpful, and friendly.
    You must answer the user's question based ONLY on the provided context.
    If the context doesn't contain the answer, you MUST state that you don't have enough information and suggest contacting human support.
    Never make up information. Be concise and clear in your answers.

    Context:
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# --- 4. Main Application Logic ---
retriever = get_retriever()
rag_chain = get_context_aware_rag_chain(retriever)

# --- 5. Chat UI Logic ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I am SentioBot. How can I assist you with your Nexora devices today?"),
    ]

for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)

if user_query := st.chat_input("Ask me about Nexora products..."):
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.write(user_query)

    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(
                {"input": user_query, "chat_history": st.session_state.chat_history}
            )
            st.write(response["answer"])
            st.session_state.chat_history.append(AIMessage(content=response["answer"]))