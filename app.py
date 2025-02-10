import streamlit as st
import os
from langchain_pinecone import PineconeVectorStore
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.helper import download_huggingface_embedding, load_data, load_data_from_uploaded_pdf, load_data_from_url, text_split
from langchain_community.vectorstores import Chroma
from src.helper import (
    download_huggingface_embedding,
    load_data_from_uploaded_pdf,
    load_data_from_url,
    text_split,
)

# Load environment variables
load_dotenv()

# Set API Key for LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set up embedding function
embeddings = download_huggingface_embedding()

# Define persistent storage for ChromaDB in Streamlit Cloud
PERSIST_DIR_DEFAULT = "/tmp/chroma_db"
PERSIST_DIR_PDF = "/tmp/chroma_db_PDF"
PERSIST_DIR_URL = "/tmp/chroma_db_URL"

# Ensure directories exist
os.makedirs(PERSIST_DIR_DEFAULT, exist_ok=True)
os.makedirs(PERSIST_DIR_PDF, exist_ok=True)
os.makedirs(PERSIST_DIR_URL, exist_ok=True)

def main():
    # Streamlit Page Config
    st.set_page_config(page_title="Medical-bot", page_icon="🩺", layout="wide")

    # Sidebar for input selection
    st.sidebar.title("Select Input Type")
    input_type = st.sidebar.radio("Choose an option:", ["Default", "URL", "PDF"], index=0)

    uploaded_file = None
    url = ""

    if input_type == "PDF":
        uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    elif input_type == "URL":
        url = st.sidebar.text_input("Enter a URL")

    # Title & User Input
    st.title("Healthcare Chatbot")
    question_input = st.text_input("Type your Question Here", "")

    # Initialize document search
    docsearch = None

    if input_type == "PDF" and uploaded_file:
        st.success(f"Processing PDF: {uploaded_file.name}")
        pdf_path = "/tmp/uploaded_file.pdf"

        # Save uploaded file
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process PDF
        docs = load_data_from_uploaded_pdf(pdf_path)
        doc_chunks = text_split(docs)

        # Create ChromaDB index
        docsearch = Chroma.from_documents(
            documents=doc_chunks,
            embedding=embeddings,
            collection_name="PDF_database",
            persist_directory=PERSIST_DIR_PDF,
        )
        st.success("Index loaded successfully")

    elif input_type == "URL" and url:
        st.success(f"Processing URL: {url}")

        # Fetch data from URL
        docs = load_data_from_url(url=url)
        doc_chunks = text_split(docs)

        # Create ChromaDB index
        docsearch = Chroma.from_documents(
            documents=doc_chunks,
            embedding=embeddings,
            collection_name="URL_database",
            persist_directory=PERSIST_DIR_URL,
        )
        st.success("Index loaded successfully")

    elif input_type == "Default":
        st.success("Using Default Medical Data")
        try:
            docsearch = Chroma(
                persist_directory=PERSIST_DIR_DEFAULT,
                collection_name="default_database",
                embedding_function=embeddings,
            )
            st.success("Default ChromaDB Index loaded!")
        except Exception as e:
            st.error(f"Error loading default index: {e}")

    # Run LLM-based QA if docsearch is ready
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

        # Initialize LLM
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="mixtral-8x7b-32768",
            temperature=0.5,
            max_tokens=1000,
            timeout=60,
        )

        # Retrieval-based Q&A
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )

        # Manage chat history
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Process user question
        if question_input:
            result = qa.invoke(question_input)
            response = result["result"]
            st.session_state["chat_history"].append((question_input, response))

        # Display chat history
        for question, answer in st.session_state["chat_history"]:
            st.write(f"**Q:** {question}")
            st.write(f"**A:** {answer}")

    else:
        st.error("No document search index available. Please select an option to proceed.")

if __name__ == "__main__":
    main()
