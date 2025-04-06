import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

@st.cache_resource
def load_qa_chain():
    # Load precomputed FAISS index and set up QA chain.
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
    
    # Load API key from Streamlit secrets
    try:
        API_KEY = st.secrets["GOOGLEAI_API_KEY"]
    except KeyError:
        st.error("Google AI API key not found in Streamlit secrets. Please configure it in secrets.toml or app settings.")
        return None
    
    # Set up Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=API_KEY)
    
    # Set up retrieval chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain

# Load the QA system
qa_chain = load_qa_chain()
if qa_chain is None:
    st.stop()

# Streamlit UI
st.title("MamaQuery")
st.write("Let me help you with your question about Mamaearth and its products!")
query = st.text_input("Enter your question:", key="query_input")
if st.button("Submit"):
    if query:
        with st.spinner("Waiting for response..."):
            try:
                result = qa_chain.run(query)
                # Handle case where result might be a string or dict
                answer = result if isinstance(result, str) else result.get("result", "No answer found")
                st.write("Answer:", answer)
            except Exception as e:
                st.error(f"Error processing query: {e}")
    else:
        st.warning("Please enter a question")
