import os
import git
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# 1. Load environment variables (for OpenAI API key)
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# 2. Clone the GitHub repo if not already present
repo_url = "https://github.com/chefgs/terraform_repo.git"
repo_dir = "/tmp/terraform_repo"
if not os.path.exists(repo_dir):
    print(f"Cloning repo {repo_url} ...")
    git.Repo.clone_from(repo_url, repo_dir)
else:
    print(f"Repo already cloned at {repo_dir}")

# 3. Load and chunk the code files
loader = DirectoryLoader(repo_dir, glob="**/*.tf",  loader_cls=TextLoader,)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print(f"Loaded {len(documents)} files, split into {len(chunks)} chunks.")

# 4. Embed and store in ChromaDB
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
retriever = vectorstore.as_retriever()

# 5. Corrective RAG prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""
You are an AI assistant for the Terraform repo. Use ONLY the following retrieved context to answer the user's question.
If the answer is not contained in the context, say "I don't know" and do not attempt to answer from your own knowledge.

If the question asks for code, and code is present in the context, return the code block as-is, properly formatted.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:
"""
)

# 6. Set up the Conversational Retrieval Chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template}
)

# 7. Terminal chat interface
def chat_with_repo():
    print("Chatbot: Hi! Ask me anything about the Terraform repo (type 'exit' to quit')")
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        result = qa_chain({"question": question})
        print("Chatbot:", result["answer"])

if __name__ == "__main__":
    chat_with_repo()