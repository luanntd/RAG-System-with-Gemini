import os
import streamlit as st
from chromadb import PersistentClient
from dotenv import load_dotenv
from urllib.parse import urlparse, urlunparse

from utils.processor import process_pdf, process_web
from utils.vector_store import create_vector_store
from utils.agent import get_query_rewriter_agent, get_web_search_agent, get_rag_agent

# --- Constants and Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_system") # Provide a default
DB_PATH = os.getenv("DB_PATH", "chroma_db")
DEFAULT_SIMILARITY_THRESHOLD = 0.7
RETRIEVER_K = 5 # Number of documents to retrieve

# --- Helper Functions ---

def initialize_session_state():
    """Initializes Streamlit session state variables if they don't exist."""
    defaults = {
        'google_api_key': GOOGLE_API_KEY,
        'history': [],
        'use_web_search': False,
        'force_web_search': False,
        'similarity_threshold': DEFAULT_SIMILARITY_THRESHOLD,
        'vector_store': None,
        'processed_documents': [],
        'chroma_client': None,
        'chroma_collection': None,
        'url_input': "",
        'clear_url_input_flag': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def normalize_url(url: str) -> str:
    """
    Normalizes a URL for consistent checking and storage.
    - Adds 'http' if no scheme is present.
    - Converts scheme and domain to lowercase.
    - Removes 'www.' prefix.
    - Removes trailing slashes from the path.
    - Removes fragments (#...).
    """
    url = url.strip()
    if not url:
        return ""

    # Add scheme if missing (default to http for parsing)
    if '://' not in url:
        url = 'http://' + url

    try:
        parts = urlparse(url)

        # Lowercase scheme and netloc (domain)
        scheme = parts.scheme.lower()
        netloc = parts.netloc.lower()

        # Remove 'www.' prefix
        if netloc.startswith('www.'):
            netloc = netloc[4:]

        # Remove trailing slashes from path, but keep root '/'
        path = parts.path.rstrip('/')
        if not path and parts.path == '/': # Keep root slash if original path was only '/'
             path = '/'
        # If path became empty after stripping and wasn't root, ensure it starts with / if netloc exists
        elif not path and parts.path != '/' and netloc:
             path = '' # Or '/' depending on desired strictness, empty seems safer.
        elif path and not path.startswith('/') and netloc:
            path = '/' + path # Ensure path starts with / if not empty

        # Reconstruct without query params and fragment for basic normalization
        # Note: Ignoring query params for simplicity here. Robust normalization might sort/handle them.
        normalized = urlunparse((scheme, netloc, path, '', '', ''))
        return normalized
    except ValueError:
        st.warning(f"⚠️ Could not properly normalize URL: {url}. Using original.")
        return url


def load_vector_store():
    """Loads or initializes the ChromaDB vector store and retrieves processed documents."""
    if st.session_state.vector_store is None:
        try:
            st.session_state.chroma_client = PersistentClient(path=DB_PATH)
            st.session_state.chroma_collection = st.session_state.chroma_client.get_or_create_collection(name=COLLECTION_NAME)

            # Wrap collection in Langchain vector store
            st.session_state.vector_store = create_vector_store(
                st.session_state.google_api_key,
                client=st.session_state.chroma_client
            )

            # Retrieve metadata (source names) of already processed documents
            results = st.session_state.chroma_collection.get(include=['metadatas'])
            if results and 'metadatas' in results and results['metadatas']:
                processed_docs = set()
                for meta in results['metadatas']:
                    if meta and 'source' in meta:
                         processed_docs.add(meta['source'])
                st.session_state.processed_documents = list(processed_docs) # Convert back to list for consistency
                st.success(f"✅ Loaded {len(st.session_state.processed_documents)} documents from database.")
            else:
                st.session_state.processed_documents = []
                st.info("ℹ️ No existing documents found in the database.")

        except Exception as e:
            st.session_state.vector_store = None
            st.session_state.processed_documents = []
            st.session_state.chroma_client = None
            st.session_state.chroma_collection = None
            st.warning(f"⚠️ Error loading/creating vector store: {e}")

def add_texts_to_vector_store(texts, source_name):
    """Adds processed text documents to the vector store."""
    if not texts:
        st.warning(f"⚠️ No text extracted from {source_name}. Skipping.")
        return False
    try:
        if st.session_state.vector_store is None:
            # Initialize vector store if it doesn't exist yet
             st.session_state.vector_store = create_vector_store(
                 st.session_state.google_api_key,
                 texts=texts, # Pass initial texts if needed by create_vector_store
                 client=st.session_state.chroma_client
             )
             # Ensure collection is updated if vector store was just created
             st.session_state.chroma_collection = st.session_state.chroma_client.get_or_create_collection(name=COLLECTION_NAME)

        else:
            st.session_state.vector_store.add_documents(texts)

        st.session_state.processed_documents.append(source_name)
        st.success(f"✅ Added source: {source_name} to the database.")
        return True
    except Exception as e:
        st.error(f"❌ Error adding {source_name} to vector store: {e}")
        return False

def clear_chat_history():
    """Clears the chat history."""
    st.session_state.history = []
    st.success("Chat history cleared.")

def clear_vector_database():
    """Clears all documents from the ChromaDB collection."""
    if st.session_state.chroma_collection:
        try:
            existing_ids = st.session_state.chroma_collection.get(include=[])['ids']
            if existing_ids:
                st.session_state.chroma_collection.delete(ids=existing_ids)
                st.session_state.processed_documents = []
                st.success("✅ Database cleared successfully. Note that this action does not delete the uploaded files in current session state.")
            else:
                st.info("ℹ️ Database is already empty.")
        except Exception as e:
            st.error(f"❌ Error clearing database: {e}")
    else:
        st.warning("⚠️ Vector store not initialized. Cannot clear database.")

def display_processed_sources():
    """Displays the list of processed documents/URLs in the sidebar."""
    if st.session_state.processed_documents:
        st.sidebar.header("📚 Processed Sources")
        for source in sorted(list(set(st.session_state.processed_documents))): # Ensure uniqueness and sort
            icon = "📄" if source.lower().endswith(".pdf") else "🌐"
            st.sidebar.text(f"{icon} {source}")

def display_chat_history():
    """Displays the chat messages from session state."""
    for chat in st.session_state.history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

def rewrite_query(query):
    """Rewrites the user query using the query rewriter agent."""
    try:
        query_rewriter = get_query_rewriter_agent()
        rewritten_query = query_rewriter.run(query).content
        # Optionally display the rewritten query
        # with st.expander("🔄 Rewritten Query"):
        #     st.write(f"Original: {query}")
        #     st.write(f"Rewritten: {rewritten_query}")
        return rewritten_query
    except Exception as e:
        st.error(f"❌ Error rewriting query: {str(e)}")
        return query

def search_documents(query):
    """Searches the vector store for relevant documents."""
    if not st.session_state.vector_store:
        st.info("ℹ️ Vector store is not available for document search.")
        return [], ""

    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": RETRIEVER_K,
            "score_threshold": st.session_state.similarity_threshold
        }
    )
    try:
        with st.spinner("Searching documents..."):
            docs = retriever.invoke(query)
            if docs:
                context = "\n\n".join([d.page_content for d in docs])
                st.info(f"📊 Found {len(docs)} relevant document chunks.")
                return docs, context
            else:
                st.info("ℹ️ No relevant documents found matching the threshold.")
                return [], ""
    except Exception as e:
        st.error(f"❌ Error searching documents: {e}")
        return [], ""

def search_web(query):
    """Searches the web using the web search agent."""
    try:
        with st.spinner("🔍 Searching the web..."):
            web_search_agent = get_web_search_agent()
            web_results = web_search_agent.run(query).content
            if web_results:
                st.info("🌐 Web search successful.")
                return f"Web Search Results:\n{web_results}"
            else:
                st.info("🕸️ Web search returned no results.")
                return ""
    except Exception as e:
        st.error(f"❌ Web search error: {str(e)}")
        return ""

def generate_response(original_query, rewritten_query, context):
    """Generates the final response using the RAG agent."""
    try:
        with st.spinner("🤖 Generating response..."):
            rag_agent = get_rag_agent()

            if context:
                full_prompt = f"""Based on the following context, answer the question.

Context:
{context}

Original Question: {original_query}
Rewritten Question (for context search): {rewritten_query}

Answer:"""
            else:
                # Fallback if no context from documents or web
                full_prompt = f"Answer the following question: {rewritten_query}"
                st.info("ℹ️ No specific context found. Answering based on general knowledge.")

            response = rag_agent.run(full_prompt)
            return response.content
    except Exception as e:
        st.error(f"❌ Error generating response: {str(e)}")
        return "Sorry, I encountered an error while generating the response."

# --- Streamlit App UI and Logic ---

def main():
    st.set_page_config(layout="wide")
    st.title("🤔 RAG System")

    initialize_session_state()
    load_vector_store()

    if st.session_state.get('clear_url_input_flag', False):
        st.session_state.url_input = ""
        st.session_state.clear_url_input_flag = False

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Controls")
        if st.button("🗑️ Clear Chat History"):
            clear_chat_history()
        if st.button("⚠️ Clear Document Database"):
            clear_vector_database()
        
        st.header("🔧 Configuration")
        st.session_state.use_web_search = st.checkbox(
            "Enable Web Search", value=st.session_state.use_web_search
        )
        st.session_state.force_web_search = st.checkbox(
            "Force Web Search", value=st.session_state.force_web_search,
            help="Always use web search, even if documents are found."
        )
        st.session_state.similarity_threshold = st.slider(
            "Document Similarity Threshold",
            min_value=0.0, max_value=1.0, value=st.session_state.similarity_threshold, step=0.05,
            help="Minimum relevance score for document retrieval (higher is stricter)."
        )

        st.header("💾 Data Input")
        uploaded_files = st.file_uploader(
            "Upload PDF Files", type=["pdf"], accept_multiple_files=True
        )
        web_url = st.text_input(
            "Enter Website URL",
            key="url_input"
        )

        display_processed_sources()

    # --- Process Uploads ---
    # Process PDFs
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            if file_name not in st.session_state.processed_documents:
                with st.spinner(f'Processing PDF: {file_name}...'):
                    texts = process_pdf(uploaded_file)
                    add_texts_to_vector_store(texts, file_name)
                 
    if web_url:
        normalized_url = normalize_url(web_url)
        if normalized_url:
            # Check if the *normalized* URL has already been processed
            if normalized_url not in st.session_state.processed_documents:
                with st.spinner(f'Processing URL: {web_url}...'):
                    # Process using the *original* URL input
                    texts = process_web(web_url)
                    if add_texts_to_vector_store(texts, normalized_url):
                        st.session_state.clear_url_input_flag = True
                        st.rerun()

    # --- Chat Interface ---
    display_chat_history()

    # Get user input
    prompt = st.chat_input("Ask a question about your documents or the web...")

    if prompt:
        # Add user message to UI and history
        st.chat_message("user").write(prompt)
        st.session_state.history.append({"role": "user", "content": prompt})

        # 1. Rewrite Query
        rewritten_query = rewrite_query(prompt)

        # 2. Search Strategy
        doc_context = ""
        web_context = ""
        docs = []

        # Try document search first unless web search is forced
        if not st.session_state.force_web_search:
            docs, doc_context = search_documents(rewritten_query)

        # Decide if web search is needed
        use_web = st.session_state.force_web_search or (st.session_state.use_web_search and not doc_context)

        if use_web:
            web_context = search_web(rewritten_query)
            if st.session_state.force_web_search and not web_context:
                 st.warning("Forced web search did not return results.")
            elif not doc_context and web_context:
                 st.info("Using web search results as fallback.")
            elif st.session_state.force_web_search and web_context:
                 st.info("Using forced web search results.")


        # 3. Combine Context (prioritize document context if available and not forcing web)
        final_context = ""
        if st.session_state.force_web_search:
            final_context = web_context # Use only web if forced
        elif doc_context:
            final_context = doc_context # Use docs if found
        elif web_context: # Use web only if docs weren't found (and web search was enabled/successful)
             final_context = web_context

        # 4. Generate Response
        assistant_response = generate_response(prompt, rewritten_query, final_context)

        # Add assistant response to UI and history
        st.chat_message("assistant").write(assistant_response)
        st.session_state.history.append({"role": "assistant", "content": assistant_response})

        # Optional: Display sources used if context came from documents
        # if not st.session_state.force_web_search and docs:
        #     with st.expander("📚 Document Sources Used"):
        #         for i, doc in enumerate(docs):
        #             source = doc.metadata.get('source', 'Unknown Source')
        #             st.write(f"**{i+1}. {source}**")
        #             st.caption(f"{doc.page_content[:250]}...") # Show snippet

if __name__ == "__main__":
    main()
