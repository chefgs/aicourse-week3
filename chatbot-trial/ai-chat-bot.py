import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Step 1: Load and extract information from the website
loader = WebBaseLoader("https://cloudenginelabs.io")
documents = loader.load()

# Step 1b: Split documents into chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)
docs = text_splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(docs)} chunks")

# Step 2: Embed the documents and create a vector store for retrieval
embeddings = OpenAIEmbeddings()
# Persist the vector store to disk for future use
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
print("Vector store created and persisted to ./chroma_db")

# Step 3: Create a retriever with search parameters
retriever = vectorstore.as_retriever(
    search_type="similarity",  # similarity, mmr, or similarity_score_threshold
    search_kwargs={"k": 5}     # number of documents to retrieve
)

# Step 4: Create a custom prompt template
custom_template = """You are an AI assistant for CloudEngineLabs. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say you don't know. Don't try to make up an answer.

Retrieved context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=custom_template
)

# Step 5: Set up memory and create the conversational RAG chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = OpenAI(temperature=0)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
    return_source_documents=True,  # Return source docs for explanation
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
        print("\nChatbot:", result["answer"])
        
        # RAG Explanation: Show source documents that were used for the answer
        if result.get("source_documents"):
            print("\nSources used:")
            for i, doc in enumerate(result["source_documents"][:3], 1):  # Show up to 3 sources
                source = doc.metadata.get('source', 'Unknown')
                print(f"  {i}. {source}")

# To start chatting, call chat_with_website()
if __name__ == "__main__":
    # Allow running the chatbot from the terminal
    chat_with_website()

