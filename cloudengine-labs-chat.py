#!/usr/bin/env python

import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Try to find and load the .env file
dotenv_path = find_dotenv()
if not dotenv_path:
    print("Warning: .env file not found!")
else:
    load_dotenv(dotenv_path)

# Verify if the API key is loaded
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
else:
    print("OPENAI_API_KEY successfully loaded from .env file")

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("CloudEngineLabs Chatbot 🤖")
st.write("Ask me anything about [cloudenginelabs.io](https://cloudenginelabs.io)!")

@st.cache_resource(show_spinner="Loading website and building vector store...")
def setup_chain():
    loader = WebBaseLoader("https://cloudenginelabs.io")
    docs = loader.load()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = OpenAI(temperature=0)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return qa_chain, memory

qa_chain, memory = setup_chain()

def get_response(question):
    result = qa_chain({"question": question})
    return result["answer"]

# Chat UI
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    answer = get_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Chatbot", answer))

# Display chat history
for speaker, text in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Chatbot:** {text}")