import streamlit as st
import os
import io
import json
from datetime import datetime
from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.api_core import exceptions

from unstructured.partition.auto import partition
from langchain.text_splitter import RecursiveCharacterTextSplitter

import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
try:
    # Attempt to get the API key from Streamlit's secrets management
    api_key = st.secrets["API_KEY"]
except (KeyError, FileNotFoundError):
    # Fallback to environment variable if not in secrets (for local development)
    api_key = os.getenv("API_KEY")

# Initialize Gemini client
if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {e}")
        client = None
else:
    client = None

# Create output directory for saving files
OUTPUT_DIR = "rag_outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Context limits (conservative estimates)
MAX_CONTEXT_CHARS = 80000
MAX_CHUNKS = 30

# File management constants
CONTEXT_FILE = "context_history.txt"
ANSWER_FILE = "answer_history.txt"
MAX_ITERATIONS = 3

# Embedding model for semantic search
embedder = SentenceTransformer('all-MiniLM-L6-v2',device='cpu')

def save_to_file(content, filename):
    """Save content to a text file in the output directory."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath
    except Exception as e:
        st.error(f"Error saving file {filename}: {e}")
        return None

def append_to_history_file(content, filename):
    """Append content to a history file and manage iterations."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Initialize session state for iteration tracking
    if 'iteration_count' not in st.session_state:
        st.session_state.iteration_count = 0
    
    # Clear file if starting new cycle
    if st.session_state.iteration_count >= MAX_ITERATIONS:
        st.session_state.iteration_count = 0
        if os.path.exists(filepath):
            os.remove(filepath)
    
    try:
        # Append to file
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Iteration {st.session_state.iteration_count + 1} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n")
            f.write(content)
            f.write("\n\n")
        
        st.session_state.iteration_count += 1
        return filepath
    except Exception as e:
        st.error(f"Error appending to file {filename}: {e}")
        return None

def delete_file(filename):
    """Delete a file from the output directory."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting file {filename}: {e}")
        return False

def get_user_files():
    """Get list of user files (excluding context and answer history files)."""
    if not os.path.exists(OUTPUT_DIR):
        return []
    
    all_files = os.listdir(OUTPUT_DIR)
    # Filter out context and answer history files
    user_files = [f for f in all_files if f not in [CONTEXT_FILE, ANSWER_FILE]]
    return sorted(user_files, reverse=True)

def generate_chunks_filename(uploaded_files):
    """Generate a descriptive filename for chunks based on uploaded files and session."""
    # Get session ID (create one if doesn't exist)
    if 'session_id' not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    session_id = st.session_state.session_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean and combine file names (remove extensions and special characters)
    file_names = []
    for file in uploaded_files:
        # Remove extension and clean filename
        clean_name = os.path.splitext(file.name)[0]
        # Replace special characters with underscores
        clean_name = "".join(c if c.isalnum() else "_" for c in clean_name)
        # Limit length to avoid overly long filenames
        clean_name = clean_name[:20]
        file_names.append(clean_name)
    
    # Combine file names (limit total length)
    combined_names = "_".join(file_names)
    if len(combined_names) > 50:
        combined_names = combined_names[:50] + "_etc"
    
    # Generate final filename
    filename = f"{combined_names}_{timestamp}_session_{session_id}.txt"
    return filename

# --- Advanced Ingestion and Chunking ---
def process_documents(uploaded_files):
    """
    Processes uploaded files using 'unstructured' for robust parsing and
    LangChain for sophisticated chunking.
    """
    all_chunks = []
    chunks_content = []  # For saving to file

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " "],
        length_function=len,
        is_separator_regex=False,
    )

    # Generate descriptive filename
    chunks_filename = generate_chunks_filename(uploaded_files)
    
    # Add metadata header
    session_id = st.session_state.get('session_id', 'unknown')
    metadata_header = f"""Document Processing Metadata
{'='*50}
Session ID: {session_id}
Processing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Files Processed: {', '.join([f.name for f in uploaded_files])}
Total Files: {len(uploaded_files)}
{'='*50}

"""
    chunks_content.append(metadata_header)

    for file in uploaded_files:
        try:
            file_content = io.BytesIO(file.read())
            file_content.name = file.name

            elements = partition(file=file_content)
            full_text = "\n\n".join([el.text for el in elements if hasattr(el, 'text') and el.text])

            if not full_text.strip():
                st.warning(f"No text content found in {file.name}")
                continue

            chunks = text_splitter.create_documents([full_text])

            # Add file separator
            chunks_content.append(f"\n{'#'*60}")
            chunks_content.append(f"FILE: {file.name}")
            chunks_content.append(f"File Size: {len(full_text)} characters")
            chunks_content.append(f"Chunks Generated: {len(chunks)}")
            chunks_content.append(f"{'#'*60}\n")

            for i, chunk in enumerate(chunks):
                chunk.metadata['source'] = file.name
                chunk.metadata['chunk_id'] = i
                chunk.metadata['session_id'] = session_id
                chunk.metadata['processing_time'] = datetime.now().isoformat()
                
                chunks_content.append(
                    f"Source: {file.name} | Chunk {i} | Session: {session_id}\n"
                    f"Content: {chunk.page_content}\n{'-'*50}"
                )

            all_chunks.extend(chunks)

        except Exception as e:
            st.error(f"Error processing file {file.name}: {e}")
            chunks_content.append(f"\nERROR processing {file.name}: {str(e)}\n")
            continue

    if chunks_content:
        chunks_text = "\n\n".join(chunks_content)
        saved_path = save_to_file(chunks_text, chunks_filename)
        if saved_path:
            st.success(f"Source chunks saved to: {chunks_filename}")
            st.info(f"Session ID: {session_id}")

    return all_chunks

# --- Semantic Retrieval Logic ---
def retrieve_and_format_chunks(query, retriever_data):
    """
    Retrieves relevant chunks using dense embeddings and formats them.
    """
    if not retriever_data:
        st.error("Retriever data is not available. Please process documents first.")
        return []

    documents = retriever_data["documents"]
    embeddings = retriever_data["embeddings"]

    # Compute query embedding
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, embeddings)[0]

    # Pick top-k
    top_k = min(MAX_CHUNKS, len(cosine_scores))
    top_k_indices = cosine_scores.argsort(descending=True)[:top_k]

    selected_chunks = []
    total_chars = 0

    for idx in top_k_indices:
        score = cosine_scores[idx].item()
        if score < 0.2:
            continue
        doc = documents[idx]
        chunk_text = (
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Chunk ID: {doc.metadata.get('chunk_id', 'Unknown')}\n"
            f"Session ID: {doc.metadata.get('session_id', 'Unknown')}\n"
            f"Similarity Score: {score:.3f}\n\n"
            f"Content: {doc.page_content}"
        )
        if total_chars + len(chunk_text) > MAX_CONTEXT_CHARS:
            break
        selected_chunks.append(chunk_text)
        total_chars += len(chunk_text)

    return selected_chunks

# --- Gemini Model Interaction (unchanged) ---
def get_gemini_response(prompt, client):
    """
    Sends a prompt to the Gemini model using the streaming API.
    """
    model_name = 'learnlm-2.0-flash-experimental'
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    is_json = "Return a single, valid JSON object" in prompt or "Return ONLY the JSON object" in prompt

    generation_config = types.GenerateContentConfig(
        response_mime_type="application/json" if is_json else "text/plain",
        temperature=0.1,
        max_output_tokens=4096
    )

    try:
        stream = client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=generation_config,
        )
        full_response = ""
        for chunk in stream:
            if chunk.text:
                full_response += chunk.text
        return full_response

    except exceptions.GoogleAPICallError as e:
        if "API_KEY_INVALID" in str(e):
            return f"Error: Invalid API key. {e}"
        if "permission" in str(e).lower() or "model not found" in str(e).lower():
            return f"Error: Permission issue for model '{model_name}'. {e}"
        return f"Google API error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

# --- Full RAG pipeline ---
def perform_rag(query, retriever_data, client):
    if not retriever_data:
        return "Error: No documents available for processing."

    selected_chunks = retrieve_and_format_chunks(query, retriever_data)
    if not selected_chunks:
        return "Error: No relevant chunks found for the query."

    # Save context to history file
    chunks_text = "\n\n" + "="*80 + "\n\n".join(selected_chunks)
    session_id = st.session_state.get('session_id', 'unknown')
    ctx_content = (
        f"Query: {query}\n"
        f"Session ID: {session_id}\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Selected {len(selected_chunks)} chunks\n\nContext:\n{chunks_text}"
    )
    append_to_history_file(ctx_content, CONTEXT_FILE)

    final_prompt = f"""
    You are an expert document analysis system. Analyze ONLY the context below:

    {chunks_text}

    User Query: "{query}"

        **Output Format:**
    INSTRUCTIONS:
• If the answer cannot be found in the context, output exactly, dont mention any chunk id:
  {{ "decision": "Information Not Found", "amount": "N/A", "justification": [] }}
• Otherwise, output exactly one JSON object with THREE keys:
  1. decision: a string
  2. amount: a number or "N/A"
  3. justification: a list of objects, each with (NOTE: It should never be empty unless no information found always give justification):
     – source: the source filename
     – clause: the exact snippet from the context
     – reasoning: why that snippet leads to the decision

Return ONLY the JSON object (no extra text, no markdown).
    """

    answer = get_gemini_response(final_prompt, client)
    
    # Save answer to history file
    ans_content = (
        f"Query: {query}\n"
        f"Session ID: {session_id}\n"
        f"Answer:\n{answer}"
    )
    append_to_history_file(ans_content, ANSWER_FILE)
    
    return answer

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Simple Working RAG System", layout="wide")
    st.title("⚙️ Simple Working RAG System")
    
    # Display session info
    if 'session_id' in st.session_state:
        st.sidebar.info(f"Session ID: {st.session_state.session_id}")

    if not client:
        st.error("Google API Key missing or invalid. Configure via .env or Streamlit secrets.")
        return

    # Sidebar: upload & process
    with st.sidebar:
        st.header("1. Upload Documents")
        uploaded = st.file_uploader(
            "Choose files", accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'md', 'pptx', 'eml']
        )

        if uploaded:
            curr_id = "".join(sorted(f.name+str(f.size) for f in uploaded))
            if st.session_state.get('processed_files_id') != curr_id:
                with st.spinner("Processing and chunking..."):
                    docs = process_documents(uploaded)
                    texts = [d.page_content for d in docs]
                    embs = embedder.encode(texts, convert_to_tensor=True)
                    st.session_state.retriever_data = {
                        "documents": docs,
                        "embeddings": embs
                    }
                    st.session_state.processed_files_id = curr_id
                    st.success(f"{len(docs)} chunks created and indexed.")

        if st.session_state.get("retriever_data"):
            st.sidebar.info(f"Context limits: Max {MAX_CHUNKS} chunks, {MAX_CONTEXT_CHARS:,} chars")

        # File Management Section
        st.header("📁 File Management")
        user_files = get_user_files()
        
        if user_files:
            st.subheader("Saved Files")
            for filename in user_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(filename)
                with col2:
                    if st.button("🗑️", key=f"delete_{filename}", help=f"Delete {filename}"):
                        if delete_file(filename):
                            #st.success(f"Deleted {filename}")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {filename}")
        else:
            st.info("No saved files")
        
        # Show iteration count
        if 'iteration_count' in st.session_state:
            remaining = MAX_ITERATIONS - st.session_state.iteration_count
            #st.info(f"Iterations until history reset: {remaining}")

    # Main: query
    st.header("2. Ask a Question")
    query = st.text_input("Enter your query:", key="query_input")

    if st.button("Process Query"):
        if not query:
            st.warning("Please enter a query.")
        elif not st.session_state.get("retriever_data"):
            st.warning("Please upload/process documents first.")
        else:
            with st.spinner("Running RAG..."):
                result = perform_rag(query, st.session_state.retriever_data, client)

                if result.startswith("Error:"):
                    st.error(result)
                else:
                    # Attempt JSON parse
                    try:
                        # strip markdown fences
                        txt = result.strip().lstrip("```json").rstrip("```").strip()
                        obj = json.loads(txt)
                        st.json(obj)
                        st.info("Answer saved to history")
                    except Exception:
                        st.subheader("Raw Response")
                        st.text_area("", result, height=300)
                        st.info("Raw answer saved to history")

if __name__ == "__main__":
    main()