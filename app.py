import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

API_KEY = st.secrets["API_KEY"]
genai.configure(api_key=API_KEY)

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=API_KEY)

# Load FAISS index
vector_store = FAISS.load_local(
    "faiss_index",
    SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

# Create retriever and QA chain
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# --- Streamlit UI ---
st.set_page_config(page_title="MamaQuery FAQ Bot")
st.title("MamaQuery – Mamaearth FAQ Bot")
st.write("This bot answers to questions related Mamaearth FAQ and product queries.")
st.write("P.S. Since this is a demo model, it is not been trained on all product faqs, but its trained on famous products.")
query = st.text_input("Ask a question related to Mamaearth:")

if query:
    with st.spinner("Fetching answer..."):
        result = qa_chain({"query": query})
        st.success(result['result'])

        with st.expander("🔍 Sources"):
            for doc in result['source_documents']:
                st.markdown(f"**• Source content:** {doc.page_content}")
