import os
import streamlit as st
from chromadb import PersistentClient
from utils.processor import process_pdf, process_web
from utils.vector_store import create_vector_store
from utils.agent import get_query_rewriter_agent, get_web_search_agent, get_rag_agent
from dotenv import load_dotenv

# Constants
load_dotenv()
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# ----------------------------------------------------Set Up----------------------------------------------------

# Streamlit App Initialization
st.title("🤔 Agentic RAG System")

# Session State Initialization
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY")
if 'history' not in st.session_state:
    st.session_state.history = []
if 'use_web_search' not in st.session_state:
    st.session_state.use_web_search = False
if 'force_web_search' not in st.session_state:
    st.session_state.force_web_search = False
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.7
if 'vector_store' not in st.session_state or 'processed_documents' not in st.session_state:
    try:
        # Initialize PersistentClient once
        client = PersistentClient(path="chroma_db")
        collection = client.get_or_create_collection(name=COLLECTION_NAME)

        # Wrap collection in Chroma vector store
        st.session_state.vector_store = create_vector_store(st.session_state.google_api_key, client=client)
        
        # Retrieve all metadata (file names)
        results = collection.get()
        if results and 'metadatas' in results:
            st.session_state.processed_documents = [
                meta.get('source', 'unknown') 
                for meta in results['metadatas'] if 'source' in meta
            ]
        else:
            st.session_state.processed_documents = []
        
        st.success(f"✅ Loaded {len(st.session_state.processed_documents)} processed documents.")

    except Exception as e:
        st.session_state.vector_store = None
        st.session_state.processed_documents = []
        st.warning(f"⚠️ No existing vector store found or error loading: {e}")


#-----------------------------------------------------SideBar Configuration-----------------------------------------------------

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.history = []

# Clear processed documents in database
if st.sidebar.button("🗑️ Clear Database"):
    if st.session_state.vector_store:
        # Retrieve all document IDs
        collection = st.session_state.vector_store._client.get_collection(name="rag_system")
        doc_ids = collection.get(ids=None)['ids']  # Get all document IDs

        if doc_ids:
            # Delete all documents
            collection.delete(ids=doc_ids)
            st.session_state.processed_documents = []  # Clear file list
            st.success("✅ All processed documents in database have been cleared. Note: This action does not delete uploaded files or web url in current state.")


# Web Search Configuration
st.sidebar.header("🌐 Web Search Configuration")
st.session_state.use_web_search = st.sidebar.checkbox("Enable Web Search Fallback", value=st.session_state.use_web_search)
st.session_state.force_web_search = st.sidebar.checkbox("Force web search", value=st.session_state.force_web_search, help="Force web search even if documents are available.")


# Search Domains Configuration (Optional)
st.sidebar.header("🎯 Search Configuration")
st.session_state.similarity_threshold = st.sidebar.slider(
    "Document Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    help="Lower values will return more documents but might be less relevant. Higher values are more strict."
)

#------------------------------------------------------Chat Workflow Setting------------------------------------------------------

#-------------------------File/URL Upload Section--------------------------

st.sidebar.header("📁 Data Upload")
uploaded_files = st.sidebar.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
web_url = st.sidebar.text_input("Or enter URL")

# Process documents
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        if file_name not in st.session_state.processed_documents:
            with st.spinner('Processing PDF...'):
                texts = process_pdf(uploaded_file)
                if texts:
                    if st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(texts)
                    else:
                        st.session_state.vector_store = create_vector_store(st.session_state.google_api_key, texts)
                    st.session_state.processed_documents.append(file_name)
                    st.success(f"✅ Added PDF: {file_name} to the database.")

            uploaded_files = None  # Clear uploaded files after processing

# Process web URL
if web_url:
    if web_url not in st.session_state.processed_documents:
        with st.spinner('Processing URL...'):
            texts = process_web(web_url)
            if texts:
                if st.session_state.vector_store:
                    st.session_state.vector_store.add_documents(texts)
                else:
                    st.session_state.vector_store = create_vector_store(st.session_state.google_api_key, texts)
                st.session_state.processed_documents.append(web_url)
                st.success(f"✅ Added URL: {web_url} to the database.")

# Display sources in sidebar
if st.session_state.processed_documents:
    st.sidebar.header("📚 Processed Sources")
    for source in st.session_state.processed_documents:
        if source.endswith('.pdf'):
            st.sidebar.text(f"📄 {source}")
        else:
            st.sidebar.text(f"🌐 {source}")

#------------------------------------Chat Interface------------------------------------

# Display previous chat history
if st.session_state.history:
    for chat in st.session_state.history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

# Chat Input
prompt = st.chat_input("Your question...")

if prompt:
    # Add user message to history
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Step 1: Rewrite the query for better retrieval
    # with st.spinner("🤔 Reformulating query..."):
    try:
        query_rewriter = get_query_rewriter_agent()
        rewritten_query = query_rewriter.run(prompt).content
        
        # with st.expander("🔄 See rewritten query"):
        #     st.write(f"Original: {prompt}")
        #     st.write(f"Rewritten: {rewritten_query}")
    
    except Exception as e:
        st.error(f"❌ Error rewriting query: {str(e)}")
        rewritten_query = prompt

    # Step 2: Choose search strategy based on force_web_search toggle
    # Try document search first
    context = ""
    docs = []
    if not st.session_state.force_web_search and st.session_state.vector_store:
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5, 
                "score_threshold": st.session_state.similarity_threshold
            }
        )
        # Try to retrieve documents based on the rewritten query
        docs = retriever.invoke(rewritten_query)
        if docs:
            context = "\n\n".join([d.page_content for d in docs])
            st.info(f"📊 Found {len(docs)} relevant documents (similarity > {st.session_state.similarity_threshold})")
        elif st.session_state.use_web_search:
            st.info("🔄 No relevant documents found in database, try web search...")


    # Step 3: Use web search if:
    # 1. Web search is forced ON, or
    # 2. No relevant documents found AND web search is enabled in settings
    if (st.session_state.force_web_search or not context) and st.session_state.use_web_search:
        with st.spinner("🔍 Searching the web..."):
            try:
                web_search_agent = get_web_search_agent()
                web_results = web_search_agent.run(rewritten_query).content
                if web_results:
                    context = f"Web Search Results:\n{web_results}"
                    if st.session_state.force_web_search:
                        st.info("ℹ️ Using web search as requested.")
                    else:
                        st.info("ℹ️ Using web search since no relevant documents were found.")
            except Exception as e:
                st.error(f"❌ Web search error: {str(e)}")

    # Step 4: Generate response using the RAG agent
    with st.spinner("🤖 Thinking..."):
        try:
            rag_agent = get_rag_agent()
            
            if context:
                full_prompt = f"""Context: {context}

                            Original Question: {prompt}
                            Rewritten Question: {rewritten_query}

                            Please provide a comprehensive answer based on the available information."""
            else:
                full_prompt = f"Original Question: {prompt}\nRewritten Question: {rewritten_query}"
                st.info("ℹ️ No relevant information found in documents or web search. Use knowledge base.")

            # Generate response
            response = rag_agent.run(full_prompt)
            
            # Add assistant response to history
            st.session_state.history.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response.content)
                
                # Show sources if available
                # if not st.session_state.force_web_search and 'docs' in locals() and docs:
                #     with st.expander("🔍 See document sources"):
                #         for i, doc in enumerate(docs, 1):
                #             source_type = doc.metadata.get("source_type", "unknown")
                #             source_icon = "📄" if source_type == "pdf" else "🌐"
                #             source_name = doc.metadata.get("file_name" if source_type == "pdf" else "url", "unknown")
                #             st.write(f"{source_icon} Source {i} from {source_name}:")
                #             st.write(f"{doc.page_content[:200]}...")

        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
