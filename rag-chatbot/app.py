import streamlit as st
from rag_engine import process_pdfs, retrieve_documents, generate_answer

st.set_page_config(page_title="Offline PDF Chatbot", page_icon="📚")
st.title("📚 Chat with your PDFs")
st.caption("Ask questions about your uploaded documents using local AI.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process PDFs", disabled=not uploaded_files):
    with st.spinner("Reading PDFs and creating embeddings..."):
        try:
            vectorstore, chunk_count = process_pdfs(uploaded_files)
            st.session_state.vectorstore = vectorstore
            st.session_state.messages = []
            st.success(f"Successfully processed {len(uploaded_files)} PDF(s) into {chunk_count} chunks.")
        except Exception as error:
            st.error(f"Processing failed: {error}")

if st.session_state.vectorstore:
    st.divider()
    st.subheader("Ask your question")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about your PDFs...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating an answer..."):
                try:
                    docs = retrieve_documents(
                        st.session_state.vectorstore, question
                    )
                    answer = generate_answer(question, docs)
                    st.markdown(answer)

                    st.markdown("**Sources**")
                    for doc in docs:
                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", 0) + 1
                        st.caption(f"{source} — Page {page}")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as error:
                    st.error(f"Could not generate an answer: {error}")
