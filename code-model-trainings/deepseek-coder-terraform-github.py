import os
import git
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain

# 1. Configuration
REPO_URL = "https://github.com/chefgs/terraform_repo.git"
REPO_DIR = "/tmp/terraform_repo"
PERSIST_DIR = "/tmp/chroma_db"
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"

print("Setting up the RAG pipeline...")

# 2. Clone the GitHub repo if not already present
if not os.path.exists(REPO_DIR):
    print(f"Cloning repo {REPO_URL}...")
    git.Repo.clone_from(REPO_URL, REPO_DIR)
else:
    print(f"Repo already cloned at {REPO_DIR}")

# 3. Load and chunk the code files
print("Loading and chunking files...")
loader = DirectoryLoader(REPO_DIR, glob="**/*.tf", loader_cls=TextLoader)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Smaller chunk size for code
    chunk_overlap=50,
    separators=["\nresource", "\nmodule", "\n", " ", ""]  # Better for Terraform code
)
chunks = splitter.split_documents(documents)
print(f"Loaded {len(documents)} files, split into {len(chunks)} chunks.")

# 4. Embed and store in ChromaDB using HuggingFace embeddings
print("Creating embeddings and vector store...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
)

# 5. Load the local model
print(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True,
    device_map="auto"  # Use GPU if available, otherwise CPU
)

# 6. Create LLM pipeline
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,  # Low temperature for more deterministic responses
    repetition_penalty=1.1
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# 7. Corrective RAG prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""
You are an AI assistant for the Terraform repo. Use ONLY the following retrieved context to answer the user's question.
If the answer is not contained in the context, say "I don't know" and do not attempt to answer from your own knowledge.

If the question asks for code, and code is present in the context, return the code block as-is in triple backticks.

Retrieved context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:
"""
)

# 8. Set up the Conversational Retrieval Chain
print("Setting up conversational chain...")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    return_source_documents=True  # Return source documents for reference
)

# 9. Terminal chat interface
def chat_with_repo():
    print("\nChatbot: Hi! Ask me anything about the Terraform repo (type 'exit' to quit)")
    print("(This is using a local DeepSeek Coder 1.3B model, so responses may take longer)")
    
    while True:
        question = input("\nYou: ")
        if question.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
            
        print("Thinking...")
        result = qa_chain({"question": question})
        answer = result["answer"]
        sources = result.get("source_documents", [])
        
        print("\nChatbot:", answer)
        
        # Optionally show sources
        if sources and len(sources) > 0:
            print("\nSources:")
            for i, doc in enumerate(sources[:3]):  # Show top 3 sources
                print(f"  {i+1}. {doc.metadata.get('source', 'Unknown source')}")

# 10. Save the fine-tuned model (if you've fine-tuned it)
def save_model(output_dir="my-terraform-code-model"):
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    chat_with_repo()
    # Uncomment to save model after fine-tuning
    # save_model()