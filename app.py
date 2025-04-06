import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import os

@st.cache_resource
def load_qa_chain():
    try:
        # looading model from hf
        model_name = "justChills/MiniLM-L6-v2-MamaQuery" 
        st.write(f"Loading model from Hugging Face: {model_name}")
        
        embedder = SentenceTransformer(model_name)
        
        st.write("Model loaded successfully!")
    
    except Exception as e:
        st.error(f"failed to load embeddings: {e}")
        return None
    try:
        index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
        vector_store = FAISS.load_local(index_path, embedder, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed loading FAISS index: {e}")
        return None
    
    API_KEY = st.secrets["GOOGLEAI_API_KEY"]
    
        
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=API_KEY)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    return qa_chain

qa_chain = load_qa_chain()
if qa_chain is None:
    st.stop()

st.title("MamaQuery")
st.write("Let me help you with your question about Mamaearth and its products!")
query = st.text_input("Enter your question:", key="query_input")
if st.button("Submit"):
    if query:
        with st.spinner("generating response...."):
            try:
                result = qa_chain.run(query)
                answer = result if isinstance(result, str) else result.get("result", "No answer found")
                st.write("Answer:", answer)
            except Exception as e:
                st.error(f"Error processing query: {e}")
    else:
        st.warning("Please enter a question")
