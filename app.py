import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

@st.cache_resource
def load_qa_chain():
    # load precomputed FAISS index and set up QA chain.
   try:
        # Load embedding model offline
        embedder = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}. Ensure 'all-MiniLM-L6-v2' is cached in ~/.cache/huggingface/hub/")
        return None
    
    # Load FAISS index
    try:
        vector_store = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return None
        
    API_KEY = st.secrets["GOOGLEAI_API_KEY"]
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=API_KEY)
    
    # setting up retrieval chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain

# Load the QA system
qa_chain = load_qa_chain()

# Streamlit UI
st.title("MamaQuery")
st.write("Let me help you with your question about Mamaearth and its products!")
query = st.text_input("Enter your question:", key="query_input")
if st.button("Submit"):
    if query:
        with st.spinner("waiting for response..."):
            result = qa_chain.run(query)
            answer = result["result"]
        st.write("Answer:", answer)
    else:
        st.warning("please enter question")
