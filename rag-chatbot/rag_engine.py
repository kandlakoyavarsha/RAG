import os
import shutil
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "chroma_db"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def process_pdfs(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(uploaded_file.getbuffer())
            temp_path = temp.name

        try:
            pages = PyPDFLoader(temp_path).load()
            for page in pages:
                page.metadata["source"] = uploaded_file.name
            documents.extend(pages)
        finally:
            os.remove(temp_path)

    if not documents:
        raise ValueError("No readable text found in the uploaded PDFs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=DB_PATH,
        collection_name="pdf_rag_collection"
    )

    return vectorstore, len(chunks)

def retrieve_documents(vectorstore, question):
    return vectorstore.similarity_search(question, k=4)

def generate_answer(question, documents):
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}, "
        f"Page: {doc.metadata.get('page', 0) + 1}\n{doc.page_content}"
        for doc in documents
    )

    response = __import__("ollama").chat(
        model="qwen2.5:1.5b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the supplied context. "
                    "If the answer is not in the context, clearly say you could not "
                    "find it in the uploaded documents. Do not make up facts."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )

    return response["message"]["content"]
