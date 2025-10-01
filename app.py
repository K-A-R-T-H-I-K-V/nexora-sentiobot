import os
import re
import pickle
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.docstore.document import Document
from langchain.tools import tool 
from langchain.agents import AgentExecutor, create_react_agent
from tools import check_order_status, check_warranty_status, create_support_ticket
from langchain.retrievers import EnsembleRetriever
from langchain import hub
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import ChatMessageHistory

# --- NEW: Import mock user database ---
from mock_db import USERS

# Set up logging to see the generated queries in the terminal (optional but helpful)
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# --- 1. Setup and Configuration ---
load_dotenv()
st.set_page_config(page_title="SentioBot | Nexora Support", page_icon="💡", layout="centered")
LOG_FILE = "analytics.log"

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
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; translateY(0); } }
            /* Style for feedback buttons */
            div[data-testid="stHorizontalBlock"] > div {
                display: flex;
                justify-content: flex-end;
                gap: 5px;
                padding-top: 10px;
            }
            div[data-testid="stHorizontalBlock"] > div > button {
                background-color: #161B22;
                border: 1px solid #30363D;
                border-radius: 5px;
                width: 40px;
                height: 40px;
            }
        </style>
    """, unsafe_allow_html=True)
load_css()

# --- NEW: Analytics Logging Function ---
def log_interaction(log_data: Dict[str, Any]):
    """Appends a dictionary of interaction data to the log file."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data) + "\n")

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
def get_agent_executor(_retriever):
    """Creates the Agent with all its tools, including the RAG chain."""
    llm = get_llm()
    retrieved_docs_for_logging = []

    @tool
    def lookup_documentation(query: str) -> str:
        """
        Use this tool to answer general questions about Nexora products, policies,
        troubleshooting guides, and technical specifications. It searches the
        official documentation. Use this for any question that does not involve
        a specific order ID, serial number, or a request for a human.
        """
        nonlocal retrieved_docs_for_logging

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
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt), 
            # MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")
        ])
        docs = _retriever.invoke(query)

        retrieved_docs_for_logging = [
            {"source": d.metadata.get('source', 'N/A'), "section": d.metadata.get('section_title', 'N/A')}
            for d in docs
        ]

        if not docs:
            return "No relevant information was found in the documentation for this query."
        
        # New detailed log of retrieved documents
        retrieved_sources = [f"{doc.metadata.get('source', 'N/A')} | Section: {doc.metadata.get('section_title', 'N/A')}" for doc in docs]
        print("\nRETRIEVED CONTEXT:")
        print(json.dumps(retrieved_sources, indent=2))
        print("="*60 + "\n")

        rag_chain = (
            {
                "context": lambda x: format_docs_with_ids(x["context"]),
                "input": lambda x: x["input"],
                # "chat_history": lambda x: x["chat_history"], # We guarantee it's here
            }
            | qa_prompt
            | structured_llm_rag
        )
        response = rag_chain.invoke({
            "context": docs, 
            "input": query, 
            # "chat_history": []
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
    # Pull the base prompt
    prompt = hub.pull("hwchase17/react-chat")
    
    # THIS IS THE FINAL, UPGRADED PROMPT WITH BETTER JUDGMENT
    # A REFINED, MORE RELIABLE PROMPT STRUCTURE

    new_prompt_template = """## Persona & Objective
You are SentioBot, a helpful and precise AI support agent for Nexora Electronics. Your primary objective is to resolve user issues by using tools, recalling conversation history, and strictly following all rules.
When you receive the user's input, it may contain a special section with their profile information (like products they own). You MUST use this information to provide more relevant, personalized answers. For example, if they own a product, prioritize troubleshooting for that specific product.

---

## Rules of Engagement
1.  **Check History First (Memory):** Before doing anything else, check the `chat_history`. If the user has already provided information (like a serial number or order ID), you MUST reuse it. Do not ask for it again.
2.  **Stop if Information is Missing:** If a tool requires specific information that you don't have (and it's not in the history), your ONLY action is to stop and ask the user for it. **Never** call a tool with placeholder information.
3.  **Be Proactive:** After a successful tool use, think about the next logical step to help the user. For example, if a warranty is active, find the claim process.
4.  **Synthesize Answers:** When you have all the necessary information (often from multiple tools), combine it into a single, comprehensive, and helpful final answer.
5.  **Handle Failures:** If `lookup_documentation` yields no relevant results, state that you couldn't find the information and ask the user if they'd like a support ticket created.
6.  **Offer the Next Action:** After successfully providing information (like the warranty claim process), if you have a tool that can perform the next logical step (like `create_support_ticket`), you MUST offer to use it.

---

## Tool Usage Protocol
* **`lookup_documentation`:** Use this FIRST for all questions about policies, product features, or troubleshooting.
* **`check_warranty_status` / `check_order_status`:** Use these ONLY when you have a specific serial number or order ID from the user or chat history.
* **`create_support_ticket`:** Use this as a LAST RESORT, either when documentation fails or when the user explicitly asks for a human agent.

---

## CRITICAL: ReAct Framework Syntax
You MUST follow this output format. Every turn must end with either `Action` or `Final Answer`.

### When to use `Action`:
Use `Action` when you need to run a tool to get more information.

Thought: I need to check the warranty. I have the serial number from the chat history. I should use the `check_warranty_status` tool.
Action: check_warranty_status
Action Input: SN-NTS-PRO-ABC123

### When to use `Final Answer`:
Use `Final Answer` for two scenarios:
1.  You have all the information needed and can directly answer the user.
2.  You need more information FROM THE USER and must ask them a question.

Thought: I have looked up the policy and see that I need a serial number. I don't have one. I must stop and ask the user.
Final Answer: To proceed with a warranty claim, I will need the serial number of your product. Could you please provide it?
""" 

    prompt.template = new_prompt_template + "\n\n" + prompt.template

    agent = create_react_agent(llm, tools, prompt) # <-- FIX 3: Use the new prompt here
    return agent, tools, lambda: retrieved_docs_for_logging

# --- 4. Main Application Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello! I am SentioBot. How can I assist you with your Nexora devices today?")]

if "store" not in st.session_state:
    st.session_state.store = ChatMessageHistory()

if "memory" not in st.session_state:
    # k=4 means it will remember the last 2 back-and-forth exchanges.
    st.session_state.memory = ConversationBufferWindowMemory(
        k=6, 
        memory_key="chat_history", # This key MUST match the placeholder in the react-chat prompt
        return_messages=True,
        chat_memory=st.session_state.store
    )
if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = None
if "last_interaction" not in st.session_state:
    st.session_state.last_interaction = {}

retriever = get_retriever()
agent, tools, get_retrieved_docs = get_agent_executor(retriever)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=st.session_state.memory, # Connect the session's memory
    verbose=True, 
    handle_parsing_errors=True
)

# --- 5. UI and Chat Logic ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

with st.sidebar:
    st.markdown("""
    <div style="text-align: center;">
        <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24" fill="none" stroke="#58A6FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
        </svg>
    </div>
    """, unsafe_allow_html=True)
    st.header("👤 User Login")

    if not st.session_state.logged_in:
        username_input = st.text_input("Username", key="username_input")
        password_input = st.text_input("Password", type="password", key="password_input")

        if st.button("Login"):
            # Check if user exists and password is correct
            # FIX 1: Convert username input to lowercase for case-insensitive matching
            user_key = username_input.lower()

            if user_key in USERS and USERS[user_key]["password"] == password_input:
                st.session_state.logged_in = True
                st.session_state.current_user_id = user_key # Use the lowercase key
                st.session_state.messages = [AIMessage(content=f"Hello {USERS[user_key]['name']}! Welcome back.")]
                st.session_state.memory.clear()
                st.rerun()
            else:
                st.error("Incorrect username or password")
    
    if st.session_state.logged_in:
        user_name = USERS[st.session_state.current_user_id]['name']
        st.success(f"Logged in as: **{user_name}**")
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.current_user_id = "guest"
            st.session_state.messages = [AIMessage(content="You have been logged out. How can I help?")]
            st.session_state.memory.clear()
            st.rerun()

    st.header("About SentioBot")
    st.info("An AI-powered assistant for Nexora Electronics, providing instant support from official documentation.")
    st.header("Tech Stack")
    st.markdown("- Streamlit\n- LangChain\n- Google Gemini\n- ChromaDB\n- BM25")
    
    # NEW FEATURE: Clear Chat History Button
    if st.button("Clear Conversation"):
        st.session_state.messages = [AIMessage(content=f"Hello {USERS[st.session_state.current_user_id]['name']}! How can I help?")]
        st.session_state.memory.clear()
        st.session_state.last_interaction = {}
        st.rerun()

st.title("SentioBot: Your Nexora Electronics Expert")

if not st.session_state.messages:
    st.session_state.messages = [AIMessage(content="Please log in to begin, or ask a question as a Guest.")]

for i, msg in enumerate(st.session_state.messages):
    avatar = USER_AVATAR if isinstance(msg, HumanMessage) else ASSISTANT_AVATAR
    with st.chat_message(msg.type, avatar=avatar):
        st.markdown(msg.content)
        # NEW: Add feedback buttons to assistant messages
        if isinstance(msg, AIMessage) and i > 0: # Don't add to the first greeting
            interaction_id = st.session_state.last_interaction.get("interaction_id")
            if interaction_id:
                cols = st.columns([10, 1, 1])
                with cols[1]:
                    if st.button("👍", key=f"thumb_up_{i}"):
                        log_interaction({"interaction_id": interaction_id, "feedback": 1})
                        st.toast("Thanks for your feedback!", icon="😊")
                with cols[2]:
                    if st.button("👎", key=f"thumb_down_{i}"):
                        log_interaction({"interaction_id": interaction_id, "feedback": -1})
                        st.toast("Thanks! We'll use this to improve.", icon="🙏")

if user_query := st.chat_input("Ask me about Nexora products..."):
    # Add user message to the display list
    st.session_state.messages.append(HumanMessage(content=user_query))
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Thinking..."):
            try:
                user_profile = USERS.get(st.session_state.current_user_id, USERS["guest"])
                if user_profile['name'] != "Guest":
                    user_profile_str = f"Name: {user_profile['name']}\nOwned Products: {', '.join(user_profile['owned_products'])}"
                    
                    combined_input = f"""
### User Profile Context
{user_profile_str}
---
### User's Question
{user_query}
"""
                else:
                    combined_input = user_query # For guests, the input is just their question

                # Now, invoke the agent with only the 'input' key.
                response = agent_executor.invoke({
                    "input": combined_input
                })
                
                final_answer = response.get("output", "I'm sorry, I encountered an error.")
                st.session_state.messages.append(AIMessage(content=final_answer))

                # NEW: Log the interaction
                interaction_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
                log_entry = {
                    "interaction_id": interaction_id,
                    "timestamp": datetime.now().isoformat(),
                    "user_id": st.session_state.current_user_id,
                    "user_query": user_query,
                    "bot_response": final_answer,
                    "retrieved_docs": get_retrieved_docs(),
                    "feedback": 0 # Default feedback
                }
                log_interaction(log_entry)
                st.session_state.last_interaction = log_entry # Store for feedback buttons
                
                st.rerun()

            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(AIMessage(content=error_msg))
                st.rerun()