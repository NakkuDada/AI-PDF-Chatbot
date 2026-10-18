
import os
import hashlib

import streamlit as st

from rag_pipeline import (
    validate_pdf,
    extract_pdf_pages,
    create_chunks,
    generate_answer,
    generate_answer_ollama,
)

from documents.vector_store import (
    index_document,
    retrieve_chunks,
)


# --------------------------------------------------
# 1. Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI PDF Chatbot")
st.caption(
    "Upload a PDF, inspect extracted text, retrieve "
    "relevant chunks, and ask questions."
)


# --------------------------------------------------
# 2. Session state
# --------------------------------------------------

if "documents" not in st.session_state:
    st.session_state.documents = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}


# --------------------------------------------------
# 3. Upload and process PDFs
# --------------------------------------------------

st.header("1. Upload PDF")

uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader",
)

if uploaded_files:
    for pdf_file in uploaded_files:
        pdf_bytes = pdf_file.getvalue()

        if not pdf_bytes:
            st.warning(f"{pdf_file.name} is empty.")
            continue

        document_id = hashlib.sha256(
            pdf_bytes
        ).hexdigest()

        # Avoid processing the same PDF again this session.
        if document_id in st.session_state.documents:
            continue

        try:
            with st.spinner(f"Processing {pdf_file.name}..."):
                validate_pdf(pdf_bytes)

                pages = extract_pdf_pages(pdf_bytes)

                if not pages:
                    st.warning(
                        f"No pages were extracted from "
                        f"{pdf_file.name}."
                    )
                    continue

                chunks = create_chunks(pages)

                if not chunks:
                    st.warning(
                        f"No text chunks were created from "
                        f"{pdf_file.name}. It may be scanned "
                        "or contain no extractable text."
                    )
                    continue

                index_document(document_id, chunks)

                st.session_state.documents[document_id] = {
                    "name": pdf_file.name,
                    "pages": pages,
                    "chunks": chunks,
                }

                st.session_state.chat_history.setdefault(
                    document_id, []
                )

            st.success(
                f"{pdf_file.name} processed and indexed successfully."
            )

        except Exception as exc:
            st.error(
                f"Could not process {pdf_file.name}: "
                f"{type(exc).__name__}: {exc}"
            )


# --------------------------------------------------
# 4. Inspect extracted text and chunks
# --------------------------------------------------

if st.session_state.documents:

    st.header("2. PDF Extraction Debugging")

    document_ids = list(st.session_state.documents.keys())

    selected_id = st.selectbox(
        "Select a processed PDF",
        options=document_ids,
        format_func=lambda doc_id: (
            st.session_state.documents[doc_id]["name"]
        ),
    )

    selected_document = st.session_state.documents[selected_id]
    pages = selected_document["pages"]
    chunks = selected_document["chunks"]

    st.write(f"**Pages extracted:** {len(pages)}")
    st.write(f"**Chunks created:** {len(chunks)}")

    with st.expander("View extracted PDF text"):
        for page in pages:
            st.markdown(f"**Page {page['page']}**")
            page_text = page.get("text", "")

            if page_text.strip():
                st.text(page_text[:2000])
            else:
                st.warning(
                    "No selectable text found on this page."
                )

    with st.expander("View first five text chunks"):
        for i, chunk in enumerate(chunks[:5]):
            st.markdown(
                f"**Chunk {i + 1} | Page {chunk['page']}**"
            )
            st.text(chunk["text"][:1000])


    # --------------------------------------------------
    # 5. Ask questions and inspect retrieval
    # --------------------------------------------------

    st.header("3. Ask Questions")

    st.caption("Ask a question about the selected PDF.")

    history = st.session_state.chat_history.setdefault(
        selected_id, []
    )

    for message in history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("error"):
                st.error(message["content"])
            else:
                st.markdown(message["content"])

            if message["role"] == "assistant":
                sources = message.get("sources", [])

                if sources:
                    with st.expander("Retrieved source chunks"):
                        for i, chunk in enumerate(sources):
                            st.markdown(
                                f"**Result {i + 1} | "
                                f"Page {chunk['page']}**"
                            )
                            st.text(chunk["text"][:1000])

    question = st.chat_input(
        "Ask something about your PDF..."
    )

    if question and question.strip():
        question = question.strip()

        history.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        try:
            # F. Retrieve relevant chunks
            with st.spinner("Retrieving relevant text..."):
                retrieved_chunks = retrieve_chunks(
                    selected_id,
                    question,
                    top_k=4,
                )

            # G. Show retrieval debugging
            with st.expander(
                f"Debug: Retrieved {len(retrieved_chunks)} chunks",
                expanded=True,
            ):
                if retrieved_chunks:
                    for i, chunk in enumerate(retrieved_chunks):
                        st.markdown(
                            f"**Result {i + 1} | "
                            f"Page {chunk['page']}**"
                        )
                        st.text(chunk["text"][:1000])
                else:
                    st.warning(
                        "No relevant chunks were retrieved."
                    )

            # H. Generate an answer using the selected provider
            if not retrieved_chunks:
                answer = (
                    "I couldn't find relevant information "
                    "in the uploaded PDF. Try a more specific "
                    "question or check the extracted text."
                )
            else:
                with st.spinner("Generating answer..."):
                    provider = os.getenv(
                        "AI_PROVIDER", "ollama"
                    ).strip().lower()

                    if provider == "ollama":
                        answer = generate_answer_ollama(
                            question,
                            retrieved_chunks,
                        )
                    elif provider == "groq":
                        answer = generate_answer(
                            question,
                            retrieved_chunks,
                        )
                    else:
                        raise ValueError(
                            "Unsupported AI_PROVIDER. "
                            "Set it to 'ollama' or 'groq'."
                        )

            if not isinstance(answer, str) or not answer.strip():
                answer = "No answer was generated."

            # I. Display and save the answer
            with st.chat_message("assistant"):
                st.markdown(answer)

                if retrieved_chunks:
                    with st.expander("Answer sources"):
                        for i, chunk in enumerate(retrieved_chunks):
                            st.markdown(
                                f"**Source {i + 1} | "
                                f"Page {chunk['page']}**"
                            )
                            st.text(chunk["text"][:1000])

            history.append({
                "role": "assistant",
                "content": answer,
                "sources": retrieved_chunks,
            })

        except Exception as exc:
            error_message = (
                f"Answer generation failed "
                f"({type(exc).__name__}): {exc}"
            )

            with st.chat_message("assistant"):
                st.error(error_message)

            history.append({
                "role": "assistant",
                "content": error_message,
                "sources": [],
                "error": True,
            })

else:
    st.info(
        "Upload a valid, text-based PDF to begin testing."
    )
