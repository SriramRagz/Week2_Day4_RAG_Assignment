
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# Application heading
st.title("📚 Multi-PDF RAG Assistant")
st.write(
    "Upload multiple PDFs and ask questions across your documents."
)

# Validate API key
if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "OpenAI API key not found. Please configure your .env file."
    )
    st.stop()

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# PDF upload section
uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# Process uploaded PDFs
if st.button("Process PDFs", type="primary"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")

    else:
        all_documents = []

        with st.spinner("Reading and processing your PDFs..."):

            try:
                for uploaded_file in uploaded_files:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    try:
                        loader = PyPDFLoader(temp_path)
                        documents = loader.load()

                        # Preserve original PDF filename
                        for document in documents:
                            document.metadata["source"] = (
                                uploaded_file.name
                            )

                        all_documents.extend(documents)

                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                if not all_documents:
                    st.error(
                        "No readable content found in the uploaded PDFs."
                    )
                    st.stop()

                # Split documents into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )

                chunks = splitter.split_documents(all_documents)

                # Generate embeddings
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small"
                )

                # Create FAISS vector database
                vector_store = FAISS.from_documents(
                    chunks,
                    embeddings
                )

                # Save vector database in session state
                st.session_state.vector_store = vector_store

                st.session_state.processed_files = [
                    file.name for file in uploaded_files
                ]

                # Display processing results
                st.success("PDFs processed successfully!")

                st.write(
                    f"**Documents uploaded:** {len(uploaded_files)}"
                )

                st.write(
                    f"**Total pages extracted:** {len(all_documents)}"
                )

                st.write(
                    f"**Total chunks created:** {len(chunks)}"
                )

            except Exception as error:
                st.error(f"Error processing PDFs: {error}")

# Question and answer section
st.divider()

st.subheader("Ask questions about your PDFs")

question = st.text_input(
    "Enter your question",
    placeholder=(
        "Example: What are the key findings across these documents?"
    )
)

if st.button("Get Answer"):

    if st.session_state.vector_store is None:
        st.warning("Please upload and process your PDFs first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "Searching documents and generating your answer..."
        ):

            try:
                # Retrieve relevant document chunks
                retriever = (
                    st.session_state.vector_store.as_retriever(
                        search_kwargs={"k": 5}
                    )
                )

                relevant_docs = retriever.invoke(question)

                # Build context with source information
                context = "\n\n".join(
                    [
                        (
                            f"Source: {doc.metadata.get('source', 'Unknown')}, "
                            f"Page: {doc.metadata.get('page', 0) + 1}\n"
                            f"{doc.page_content}"
                        )
                        for doc in relevant_docs
                    ]
                )

                # Prompt template
                prompt = ChatPromptTemplate.from_template("""
                You are a document question-answering assistant.

                Answer the question using only the provided context.

                If the documents contain conflicting information,
                clearly explain the differences and identify the sources.

                If the answer is not available in the context,
                say that the documents do not provide enough information.

                Context:
                {context}

                Question:
                {question}

                Answer:
                """)

                # Initialize language model
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0
                )

                # Generate answer
                chain = prompt | llm

                response = chain.invoke({
                    "context": context,
                    "question": question
                })

                # Display answer
                st.markdown("### Answer")
                st.write(response.content)

                # Display unique sources
                st.markdown("### Sources")

                seen_sources = set()

                for doc in relevant_docs:

                    source = doc.metadata.get(
                        "source",
                        "Unknown"
                    )

                    page = (
                        doc.metadata.get("page", 0) + 1
                    )

                    source_key = (source, page)

                    if source_key not in seen_sources:

                        st.write(
                            f"- **{source}** — Page {page}"
                        )

                        seen_sources.add(source_key)

            except Exception as error:
                st.error(
                    f"Error generating answer: {error}"
                )

# Sidebar: processed PDF list
if st.session_state.processed_files:

    st.sidebar.subheader("Processed PDFs")

    for filename in st.session_state.processed_files:
        st.sidebar.write(f"📄 {filename}")
