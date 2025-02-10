import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from chromadb.config import Settings  # ✅ Ensure DuckDB is used

from src.helper import (
    download_huggingface_embedding,
    load_data_from_uploaded_pdf,
    load_data_from_url,
    text_split,
)

def main():
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    embeddings = download_huggingface_embedding()

    # ✅ Explicitly configure ChromaDB to use DuckDB instead of SQLite
    CHROMA_SETTINGS = Settings(
        chroma_db_impl="duckdb",  # ✅ Forces DuckDB instead of SQLite
        persist_directory="/tmp/chroma_db",  # ✅ Ensures storage is inside /tmp/
    )

    CHROMA_PDF_DB = "/tmp/chroma_db_pdf"
    CHROMA_URL_DB = "/tmp/chroma_db_url"

    st.set_page_config(page_title="Medical-bot", page_icon="H", layout="wide")

    col1, col2 = st.columns([1, 3])  # Sidebar for input selection

    with col1:
        st.sidebar.title("Select Input Type")
        input_type = st.sidebar.radio("Choose an option:", ["Default", "URL", "PDF"], index=0)

        uploaded_file = None
        url = ""

        if input_type == "PDF":
            uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
        elif input_type == "URL":
            url = st.sidebar.text_input("Enter a URL")

    with col2:
        st.title("Healthcare Chatbot")
        question_input = st.text_input("Type your Question Here", "")

    # Initialize docsearch
    docsearch = None

    if input_type == "PDF" and uploaded_file:
        st.success(f"Processing PDF: {uploaded_file.name}")
        pdf_path = "/tmp/uploaded_file.pdf"  # ✅ Store uploaded files in /tmp/
        
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        docs = load_data_from_uploaded_pdf(pdf_path)
        doc_chunks = text_split(docs)
        docsearch = Chroma.from_documents(
            documents=doc_chunks,
            embedding=embeddings,
            collection_name="PDF_database",
            persist_directory=CHROMA_PDF_DB,
            client_settings=CHROMA_SETTINGS,  # ✅ Forces DuckDB instead of SQLite
        )
        st.success("Index loaded successfully")

    elif input_type == "URL" and url:
        st.success(f"Processing URL: {url}")
        docs = load_data_from_url(url=url)
        doc_chunks = text_split(docs)
        docsearch = Chroma.from_documents(
            documents=doc_chunks,
            embedding=embeddings,
            collection_name="URL_database",
            persist_directory=CHROMA_URL_DB,
            client_settings=CHROMA_SETTINGS,  # ✅ Forces DuckDB instead of SQLite
        )
        st.success("Index loaded successfully")

    elif input_type == "Default":
        st.success("Using Medical Book")
        try:
            docsearch = Chroma(
                persist_directory="/tmp/chroma_db",
                embedding_function=embeddings,
                collection_name="medical_chatbot",
                client_settings=CHROMA_SETTINGS,  # ✅ Forces DuckDB instead of SQLite
            )
            st.success("Index loaded successfully!")
        except Exception as e:
            st.error(f"Error loading default index: {e}")

    if docsearch is not None:
        prompt_template = """
        Use the given information context to provide an appropriate answer for the user's question.
        If you don't know the answer, just say you don't know. Don't make up an answer.
        Context: {context}
        Question: {question}
        Only return the answer.
        Helpful answer:
        """

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": PROMPT}

        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="mixtral-8x7b-32768",
            temperature=0.5,
            max_tokens=1000,
            timeout=60,
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs,
        )

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        if question_input:
            result = qa.invoke(question_input)
            response = result["result"]
            st.session_state["chat_history"].append((question_input, response))

        for question, answer in st.session_state["chat_history"]:
            st.write(f"**Q:** {question}")
            st.write(f"**A:** {answer}")
    else:
        st.error("No document search index available. Please select an option to proceed.")

if __name__ == "__main__":
    main()
