import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import pathlib

# Check for .streamlit/secrets.toml (Streamlit Cloud or local dev)
openai_api_key = None
secrets_toml_path = pathlib.Path(".streamlit/secrets.toml")
if secrets_toml_path.exists() and "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()
    openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in .streamlit/secrets.toml or .env file.")
    st.stop()

# --- SIDEBAR: About & Use Cases ---
st.sidebar.title("About This Chatbot 🤖")
st.sidebar.markdown(
    """
    **CloudEngineLabs Chatbot**  
    Powered by LLMs and retrieval over <https://cloudenginelabs.io>.

    **What can this bot do?**
    - Answer questions about CloudEngineLabs, its services, and website content.
    - Summarize or explain sections of the website.
    - Help you explore DevOps and automation topics discussed on CloudEngineLabs.
    - Provide quick, conversational answers with context from the website.
    """
)

# --- Main App Title ---
st.title("CloudEngineLabs Chatbot")

st.markdown(
    """
    Ask me anything about [cloudenginelabs.io](https://cloudenginelabs.io)!
    Type your question below and I'll answer using the latest content from the website.
    """
)

# --- Load website and create QA chain (cache for speed) ---
@st.cache_resource(show_spinner="Loading and indexing website...")
def setup_chain():
    loader = WebBaseLoader("https://cloudenginelabs.io")
    docs = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    # See migration guide for memory: https://python.langchain.com/docs/versions/migrating_memory/
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return qa_chain

qa_chain = setup_chain()

# --- Initialize session state for chat ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- User input box at the bottom ---
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "", placeholder="Ask a question about cloudenginelabs.io")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    # Custom response for 'who are you?'
    if user_input.strip().lower() in ["who are you?", "who are you", "what are you?", "what is your role?"]:
        answer = (
            "I am the CloudEngineLabs Chatbot, designed specifically to help you explore and understand the content, services, and topics available on https://cloudenginelabs.io. "
            "I can answer questions, summarize information, and guide you through DevOps and automation resources found on the CloudEngineLabs website."
        )
    else:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"question": user_input})
            answer = result["answer"]
    
    # Add to chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", answer))
    
    # Force a rerun to update the UI immediately
    st.rerun()

# --- Chat history display (response box above) ---
if st.session_state.chat_history:
    st.markdown("#### Chat History")
    # Show chat history in pairs (Q then A), newest at top, oldest at bottom
    history = st.session_state.chat_history
    # Reverse in steps of 2 (Q, A pairs)
    for i in range(len(history) - 2, -1, -2):
        speaker_q, text_q = history[i]
        speaker_a, text_a = history[i+1]
        if speaker_q == "You":
            st.markdown(
                f"""
                <div style='background:#1565c0;padding:10px 14px;border-radius:8px;margin-bottom:2px'>
                    <strong style='color:#fff;'>You:</strong> <span style='color:#e3f2fd'>{text_q}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        if speaker_a == "Bot":
            # Format answer as two-line paragraphs and add CTA
            paragraphs = text_a.strip().split("\n")
            formatted = "<br><br>".join([p.strip() for p in paragraphs if p.strip()])
            cta = ("<div style='margin-top:10px;'><a href='https://cloudenginelabs.io/#services' "
                   "target='_blank' style='color:#1976d2;text-decoration:underline;font-weight:bold;'>"
                   "👉 Check out CloudEngineLabs services</a></div>")
            st.markdown(
                f"""
                <div style='background:#fffde7;padding:10px 14px;border-radius:8px;margin-bottom:8px'>
                    <strong style='color:#f9a825;'>Bot:</strong> <span style='color:#6d4c41'>{formatted}</span>
                    {cta}
                </div>
                """,
                unsafe_allow_html=True
            )
