import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory

# Step 1: Load and extract information from the website
loader = WebBaseLoader("https://cloudenginelabs.io")
docs = loader.load()

# Step 2: Embed the documents and create a vector store for retrieval
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 3: Set up conversational memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Step 4: Create a conversational retrieval chain
llm = OpenAI(temperature=0)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
)

# Step 5: Define a simple chat loop
def chat_with_website():
    print("Chatbot: Hi! Ask me anything about cloudenginelabs.io (type 'exit' to quit)")
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        result = qa_chain({"question": question})
        print("Chatbot:", result["answer"])

# To start chatting, call chat_with_website()
if __name__ == "__main__":
    # Allow running the chatbot from the terminal
    chat_with_website()

