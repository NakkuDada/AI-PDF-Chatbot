
import hashlib
import pymupdf
import streamlit as st
from document_processor import extract_pdf_pages

from rag_pipeline import (
    extract_pdf_pages,
    create_chunks,
    index_document,
    retrieve_chunks,
    generate_answer
)

# STEP 1: Configure the Streamlit page
st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI PDF Chatbot")
st.write(
    "Upload a PDF and ask questions about its content. "
    "Answers include retrieved source passages."
)

# STEP 2: Initialize session state
if "documents" not in st.session_state:
    st.session_state.documents = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

# STEP 3: Upload PDF files
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

# STEP 4: Extract text and process uploaded PDFs
if uploaded_files:
    for pdf_file in uploaded_files:
        pdf_bytes = pdf_file.getvalue()

        document_id = hashlib.sha256(
            pdf_bytes
        ).hexdigest()

        if document_id not in st.session_state.documents:
            try:
                with st.spinner(
                    f"Processing {pdf_file.name}..."
                ):
                    # Extract text using PyMuPDF
                    doc = pymupdf.open(
                        stream=pdf_bytes,
                        filetype="pdf"
                    )

                    text = "\n".join(
                        page.get_text()
                        for page in doc
                    )

                    st.write("PDF:", pdf_file.name)
                    st.write(
                        "Extracted characters:",
                        len(text)
                    )

                    with st.expander(
                        f"Preview extracted text: {pdf_file.name}"
                    ):
                        if text.strip():
                            st.text_area(
                                "Extracted PDF text",
                                text[:5000],
                                height=300,
                                key=f"text_preview_{document_id}"
                            )
                        else:
                            st.warning(
                                "No selectable text was found. "
                                "This PDF may require OCR."
                            )

                    doc.close()

                    # Create chunks for searching
                    pages = extract_pdf_pages(pdf_bytes)
                    chunks = create_chunks(pages)

                    # Index chunks in the search system
                    index_document(document_id, chunks)

                    # Store processed document
                    st.session_state.documents[document_id] = {
                        "name": pdf_file.name,
                        "pages": len(pages),
                        "chunks": chunks
                    }

                    st.session_state.chat_history[
                        document_id
                    ] = []

            except Exception as exc:
                st.error(
                    f"Could not process {pdf_file.name}: {exc}"
                )

# STEP 5: Select a processed document
if st.session_state.documents:
    document_ids = list(
        st.session_state.documents.keys()
    )

    selected_id = st.selectbox(
        "Select a document",
        options=document_ids,
        format_func=lambda doc_id: (
            st.session_state.documents[doc_id]["name"]
        )
    )

    document = st.session_state.documents[selected_id]

    st.success(
        f"Ready: {document['name']} | "
        f"{document['pages']} pages | "
        f"{len(document['chunks'])} chunks"
    )

    st.divider()
    st.subheader("Ask questions")

    # STEP 6: Display previous chat history
    for message in st.session_state.chat_history[selected_id]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View source passages"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**PDF Page {source.get('page', 'Unknown')}**"
                        )
                        st.write(
                            source.get("text", "No text available")
                        )

    # STEP 7: Ask questions about the selected PDF
    question = st.chat_input(
        "What technical skills are mentioned in this PDF?"
    )

    if question:
        st.session_state.chat_history[selected_id].append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner(
                    "Searching the PDF and generating an answer..."
                ):
                    # STEP 7A: Retrieve PDF chunks
                    sources = retrieve_chunks(
                        selected_id,
                        question,
                        top_k=4
                    )

                    # STEP 7B: DEBUG - Show retrieved chunks
                    st.subheader("Debug: Retrieved PDF Chunks")

                    st.write(
                        "DEBUG: Retrieved chunks:",
                        len(sources)
                    )

                    for i, source in enumerate(sources):
                        st.write(f"Chunk {i + 1}")

                        st.write(
                            "Page:",
                            source.get("page", "Unknown")
                        )

                        st.write(
                            "Text:",
                            source.get("text", "No text")
                        )

                    # STEP 7C: Generate answer
                    answer, sources = generate_answer(
                        question,
                        sources
                    )

                # STEP 7D: Display chatbot answer
                st.subheader("Chatbot Answer")
                st.markdown(answer)

                # STEP 7E: Display source passages
                if sources:
                    with st.expander("View source passages"):
                        for source in sources:
                            st.markdown(
                                f"**PDF Page {source.get('page', 'Unknown')}**"
                            )
                            st.write(
                                source.get("text", "No text available")
                            )
                else:
                    st.warning(
                        "No source passages were retrieved. "
                        "Check the debug output above."
                    )

                # STEP 7F: Save assistant response
                st.session_state.chat_history[
                    selected_id
                ].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as exc:
                st.error(
                    f"Unable to generate an answer: {exc}"
                )

else:
    st.info("Upload a PDF to start chatting.")


    
