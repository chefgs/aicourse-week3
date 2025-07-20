# CloudEngineLabs Website Chatbot

A Streamlit-powered chatbot that answers questions about [CloudEngineLabs.io](https://cloudenginelabs.io) using the latest content from the website. Built with LangChain, OpenAI, and FAISS for efficient, context-aware Q&A.

## Features

- Conversational Q&A about CloudEngineLabs website content
- Uses OpenAI embeddings and LLM for accurate answers
- Fast retrieval with FAISS vector store
- Remembers chat history for context
- Custom response for "Who are you?" questions
- Readable, styled chat bubbles and a call-to-action link in every answer

## How It Works

1. Loads and embeds website content
2. Stores embeddings in a FAISS vector store
3. Uses a conversational retrieval chain to answer user questions
4. Maintains chat history for context-aware responses
5. Displays answers as readable paragraphs with a CTA

## Setup

1. **Clone the repository**
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3. **Create a `.env` file** with your OpenAI API key:
    ```
    OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```
4. **Run the app**
    ```bash
    streamlit run website-chatbot.py
    ```

## Customization

- To change the website, update the URL in `setup_chain()`.
- To use a different LLM or embedding model, adjust the imports and initialization.

## Concepts Used

- **Streamlit UI**: For interactive web-based chat.
- **LangChain**: For document loading, embedding, vector storage, and conversational retrieval.
- **Session State**: To persist chat history across reruns.
- **Caching**: To avoid redundant computation and speed up the app.
- **Custom Responses**: For specific questions like "Who are you?"
- **HTML/CSS Styling**: For visually distinct chat bubbles and CTA links.
- **Environment Management**: Securely loading API keys from `.env`.

## License

MIT

---

**Enjoy exploring CloudEngineLabs with your own AI-powered chatbot!**
