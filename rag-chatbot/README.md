# Offline PDF Chatbot using Retrieval-Augmented Generation (RAG)

An offline PDF chatbot that allows users to upload PDF documents and ask questions about their contents. The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and generate answers using a locally running Large Language Model (LLM).

## Features

- Upload one or multiple PDF documents.
- Extract text from PDF files.
- Split extracted text into smaller chunks.
- Generate embeddings using Sentence Transformers.
- Store document chunks and embeddings in ChromaDB.
- Retrieve relevant document chunks using semantic similarity search.
- Generate context-based answers using a locally running LLM through Ollama.
- Display source PDF names and page numbers.
- Run locally without requiring an external AI API.
- Supports offline usage after the required models and dependencies have been downloaded.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | User interface |
| LangChain | Document loading and text splitting |
| PyPDF | PDF text extraction |
| Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2 | Embedding model |
| ChromaDB | Vector database |
| Ollama | Running the local LLM |
| Qwen 2.5 (1.5B) | Answer generation |

## System Architecture

The application follows two main stages:

### 1. Document Ingestion

1. The user uploads PDF documents.
2. Text is extracted from the PDFs.
3. Extracted text is divided into smaller chunks using `RecursiveCharacterTextSplitter`.
4. The embedding model converts each chunk into a numerical vector.
5. The chunks, embeddings and metadata are stored in ChromaDB.

### 2. Question Answering

1. The user enters a question through the Streamlit interface.
2. The embedding model converts the question into a vector.
3. ChromaDB performs similarity search to retrieve the four most relevant chunks.
4. The retrieved chunks are passed to the Qwen LLM along with the user's question.
5. The LLM generates an answer using the supplied context.
6. The application displays the answer along with the source PDF names and page numbers.

## Project Structure

```text
rag-chatbot/
│
├── app.py                 # Streamlit application and user interface
├── rag_engine.py          # PDF processing, embeddings,
│                          # retrieval and answer generation
├── chroma_db/             # Persisted ChromaDB vector database
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## Prerequisites

Make sure the following are installed:

- Python 3.10 or later
- Conda or Python virtual environment
- Ollama
- Required Python libraries

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd rag-chatbot
```

### 2. Create and activate a virtual environment

Using Conda:

```bash
conda create -n ragchat python=3.10
conda activate ragchat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and configure Ollama

Install Ollama from:

https://ollama.com/

Download the Qwen model:

```bash
ollama pull qwen2.5:1.5b
```

### 5. Download the embedding model

The application uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model is downloaded automatically when first initialized, provided internet access is available. Once downloaded and cached locally, it can be used offline.

### 6. Run the application

Make sure Ollama is running, then execute:

```bash
python -m streamlit run app.py
```

To run with offline mode enabled after downloading all required models:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser.

## How to Use

1. Launch the application.
2. Upload one or more PDF documents.
3. Click **Process PDFs**.
4. Wait for document extraction, chunking, embedding generation and vector storage to complete.
5. Enter your question in the chat input.
6. The application retrieves relevant information and generates an answer.
7. View the answer and its source document pages.

## Configuration

The current implementation uses the following settings:

| Parameter | Value |
|---|---|
| Chunk size | 800 characters |
| Chunk overlap | 150 characters |
| Number of retrieved chunks (Top-K) | 4 |
| Embedding model | all-MiniLM-L6-v2 |
| Embedding dimensions | 384 |
| LLM | qwen2.5:1.5b |
| Vector database | ChromaDB |
| Execution device | CPU |

## Limitations

- The current implementation supports text-based PDFs. Scanned PDFs requiring OCR are not supported.
- Answer quality depends on the relevance of retrieved chunks and the capabilities of the local LLM.
- Similarity search may retrieve irrelevant chunks for some questions.
- The application currently replaces the existing vector database when new PDFs are processed.
- The application does not implement advanced retrieval techniques such as hybrid search or reranking.

## Future Enhancements

- Add OCR support for scanned PDFs.
- Implement hybrid search and reranking.
- Improve retrieval accuracy using metadata filtering.
- Add persistent database loading across application restarts.
- Support more local LLM models.
- Add document management features for adding and deleting PDFs.
- Implement automated evaluation of retrieval and answer quality.

## Conclusion

This project demonstrates the implementation of an offline Retrieval-Augmented Generation system that combines document processing, semantic search, vector storage and local LLM-based answer generation. It enables users to interact with their own PDF documents without depending on external AI APIs.

**The application is designed to perform document retrieval and answer generation locally, helping maintain document privacy and reducing dependence on cloud-based services.**