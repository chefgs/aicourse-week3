import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory

# Load .env variables (API keys, etc.)
load_dotenv()

# --- SIDEBAR: About & Use Cases ---
st.sidebar.title("About This Chatbot 🤖")
st.sidebar.markdown(
    """
    **POC Website Chatbot**  
    Powered by LLMs and retrieval over any website you choose.

    **What can this bot do?**
    - Answer questions about the website you provide.
    - Summarize or explain sections of the website.
    - Let you choose how detailed the answers should be.
    - Provide quick, conversational answers with context from the website.
    """
)

# --- Main App Title ---
st.title("Website Chatbot (POC)")

st.markdown(
    """
    Enter a website URL, select the level of information, and ask your questions!
    """
)

# --- User input for website and info level ---
with st.form("setup_form"):
    website_url = st.text_input("Website URL", value="https://cloudenginelabs.io")
    info_level = st.selectbox(
        "Level of Information",
        ["Short answer (summary)", "Detailed answer (more context)"],
        index=0
    )
    setup_submitted = st.form_submit_button("Load Website")

# --- Cache QA chain for each website and info level ---
@st.cache_resource(show_spinner="Loading and indexing website...", max_entries=3)
def setup_chain(url, info_level):
    loader = WebBaseLoader(url)
    # For POC: Only load a small number of docs for speed
    docs = loader.load()[:3] if info_level == "Short answer (summary)" else loader.load()[:8]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Use a fast, deterministic model for POC
    llm = OpenAI(temperature=0, max_tokens=256 if info_level == "Short answer (summary)" else 512)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return qa_chain

if setup_submitted or "qa_chain" not in st.session_state:
    st.session_state.qa_chain = setup_chain(website_url, info_level)
    st.session_state.chat_history = []

qa_chain = st.session_state.qa_chain

# --- User input and Chatbot response UI ---
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "", placeholder="Ask a question about the website")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    result = qa_chain({"question": user_input})
    answer = result["answer"]
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", answer))

# --- Chat history display ---
st.markdown("#### Chat History")
for speaker, text in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"<div style='background:#e6f7ff;padding:8px 12px;border-radius:8px;margin-bottom:4px'><strong>You:</strong> {text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background:#f6ffe6;padding:8px 12px;border-radius:8px;margin-bottom:8px'><strong>Bot:</strong> {text}</div>", unsafe_allow_html=True)