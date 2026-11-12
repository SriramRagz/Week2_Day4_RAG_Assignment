# Multi-PDF RAG Assistant

## Project Overview

The Multi-PDF RAG Assistant is a Python-based application built using Streamlit and LangChain. It allows users to upload multiple PDF documents, process their content, and ask questions across all uploaded documents using Retrieval-Augmented Generation (RAG).

The application retrieves relevant information from the uploaded documents and generates contextual answers with source filenames and page numbers.

## Features

- Upload multiple PDF documents simultaneously.
- Extract text from PDF documents.
- Split document content into manageable chunks.
- Generate embeddings using OpenAI Embeddings.
- Store document vectors in a FAISS vector database.
- Retrieve relevant information across multiple PDFs.
- Generate contextual answers using an OpenAI language model.
- Display source filenames and page numbers.
- Identify conflicting information across documents.
- Maintain the vector database using Streamlit session state.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | User interface |
| LangChain | RAG workflow orchestration |
| OpenAI Embeddings | Text embeddings |
| GPT-4o-mini | Answer generation |
| FAISS | Vector database |
| PyPDF | PDF document extraction |

## Project Structure

```text
Week2_Day4_RAG_Assignment/
│
├── rag_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
└── venv/
```

**Note:** `.env` and `venv/` are excluded from GitHub through `.gitignore`.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Week2_Day4_RAG_Assignment
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the OpenAI API key

Create a `.env` file in the project directory:

```env
OPENAI_API_KEY=your_openai_api_key
```

Never upload your actual API key to GitHub.

### 5. Run the application

```bash
streamlit run rag_app.py
```

Open the local URL displayed in your terminal.

## Testing

The application was tested using two sample PDF documents containing related project information.

**Test scenario:** Identify the recommended city for the next expansion and determine whether both documents agree.

**Results:**
- Document A recommended Hyderabad.
- Document B recommended Coimbatore.
- The application identified the conflicting recommendations.
- Source filenames and page numbers were displayed.
- Duplicate source entries were removed.

## Key Learning Outcomes

- Understanding Retrieval-Augmented Generation (RAG).
- Working with multiple PDF documents.
- Document loading and text chunking.
- Generating and storing vector embeddings.
- Implementing semantic retrieval using FAISS.
- Integrating LangChain with OpenAI models.
- Generating answers with source references.
- Handling conflicting information across documents.

## Future Enhancements

- Support additional document formats.
- Add persistent vector database storage.
- Improve retrieval accuracy through configurable chunking.
- Add document-level filtering and retrieval controls.

---

**Assignment:** Week 2 – Day 4  
**Project:** Build a Multi-PDF RAG App with Sources