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

# File management constants
CONTEXT_FILE = "context_history.txt"
ANSWER_FILE = "answer_history.txt"
MAX_ITERATIONS = 3

# Default token limits and chunk settings
DEFAULT_TOKEN_LIMIT = 500000  # For chunking decision during document processing
DEFAULT_LLM_TOKEN_LIMIT = 900000  # For semantic search decision during RAG
DEFAULT_MAX_CONTEXT_CHARS = 80000
DEFAULT_MAX_CHUNKS = 30
DEFAULT_CHUNK_SIZE = 1500  # Characters per chunk
DEFAULT_CHUNK_OVERLAP = 300  # Character overlap between chunks

# Conditional embedding model loading
@st.cache_resource
def load_embedder():
    """Load embedding model only when needed."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def get_token_limits():
    """Get current token limits and chunk settings from session state."""
    return {
        'token_limit': st.session_state.get('token_limit', DEFAULT_TOKEN_LIMIT),
        'llm_token_limit': st.session_state.get('llm_token_limit', DEFAULT_LLM_TOKEN_LIMIT),
        'max_context_chars': st.session_state.get('max_context_chars', DEFAULT_MAX_CONTEXT_CHARS),
        'max_chunks': st.session_state.get('max_chunks', DEFAULT_MAX_CHUNKS),
        'chunk_size': st.session_state.get('chunk_size', DEFAULT_CHUNK_SIZE),
        'chunk_overlap': st.session_state.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP)
    }

def initialize_session():
    """Initialize session and clean up files from other sessions."""
    # Get or create session ID
    if 'session_id' not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.session_initialized = True
        st.session_state.last_processed_files = None  # Track last processed files
        
        # Initialize token limits and chunk settings in session state
        st.session_state.token_limit = DEFAULT_TOKEN_LIMIT
        st.session_state.llm_token_limit = DEFAULT_LLM_TOKEN_LIMIT
        st.session_state.max_context_chars = DEFAULT_MAX_CONTEXT_CHARS
        st.session_state.max_chunks = DEFAULT_MAX_CHUNKS
        st.session_state.chunk_size = DEFAULT_CHUNK_SIZE
        st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP
        
        # Clean up files from other sessions
        cleanup_other_session_files()

def cleanup_other_session_files():
    """Remove files that don't belong to the current session."""
    if not os.path.exists(OUTPUT_DIR):
        return
    
    current_session = st.session_state.session_id
    files_to_delete = []
    
    for filename in os.listdir(OUTPUT_DIR):
        # Skip history files
        if filename in [CONTEXT_FILE, ANSWER_FILE]:
            continue
            
        filepath = os.path.join(OUTPUT_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Check if file contains session ID info
                if "Session ID:" in content:
                    # Extract session ID from file content
                    for line in content.split('\n'):
                        if line.startswith("Session ID:"):
                            file_session = line.split("Session ID:")[1].strip()
                            if file_session != current_session:
                                files_to_delete.append(filename)
                            break
                else:
                    # If no session ID found, it's from an old version - delete it
                    files_to_delete.append(filename)
        except Exception:
            # If we can't read the file, delete it
            files_to_delete.append(filename)
    
    # Delete files from other sessions
    for filename in files_to_delete:
        try:
            os.remove(os.path.join(OUTPUT_DIR, filename))
        except Exception:
            pass

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

def get_file_stats(filename):
    """Get character and token count for a file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        char_count = len(content)
        # Estimate tokens using word count (rough approximation: 1 token ≈ 0.75 words)
        word_count = len(content.split())
        estimated_tokens = int(word_count * 1.33)  # More conservative estimate
        
        return char_count, estimated_tokens
    except Exception as e:
        return 0, 0

def estimate_tokens(text):
    """Estimate token count for text."""
    word_count = len(text.split())
    return int(word_count * 1.33)

def get_user_files():
    """Get list of user files (excluding context and answer history files)."""
    if not os.path.exists(OUTPUT_DIR):
        return []
    
    all_files = os.listdir(OUTPUT_DIR)
    # Filter out context and answer history files
    user_files = [f for f in all_files if f not in [CONTEXT_FILE, ANSWER_FILE]]
    return sorted(user_files, reverse=True)

def create_file_signature(uploaded_files):
    """Create a unique signature for the uploaded files batch."""
    if not uploaded_files:
        return None
    
    # Create signature based on filenames and sizes
    file_info = []
    for file in uploaded_files:
        file_info.append(f"{file.name}:{file.size}")
    
    return "|".join(sorted(file_info))

def generate_filename(uploaded_files, is_chunked=False):
    """Generate a clean filename based only on document names."""
    if not uploaded_files:
        return "unknown.txt"
    
    # Clean and combine file names (remove extensions and special characters)
    file_names = []
    for file in uploaded_files:
        # Remove extension and clean filename
        clean_name = os.path.splitext(file.name)[0]
        # Replace special characters with underscores
        clean_name = "".join(c if c.isalnum() else "_" for c in clean_name)
        # Limit length to avoid overly long filenames
        clean_name = clean_name[:30]
        file_names.append(clean_name)
    
    # Combine file names (limit total length)
    combined_names = "_".join(file_names)
    if len(combined_names) > 80:
        combined_names = combined_names[:80] + "_etc"
    
    # Add chunked suffix if needed
    if is_chunked:
        filename = f"{combined_names}_chunked.txt"
    else:
        filename = f"{combined_names}.txt"
    
    return filename

# --- Advanced Ingestion and Chunking ---
def process_documents(uploaded_files):
    """
    Processes uploaded files - extracts text and chunks based on enhanced logic.
    """
    if not uploaded_files:
        return []
    
    # Get current token limits and chunk settings
    limits = get_token_limits()
    token_limit = limits['token_limit']
    llm_token_limit = limits['llm_token_limit']
    chunk_size = limits['chunk_size']
    chunk_overlap = limits['chunk_overlap']
    
    # Create file signature to prevent duplicate processing
    file_signature = create_file_signature(uploaded_files)
    
    # Check if we've already processed these exact files
    if st.session_state.get('last_processed_files') == file_signature:
        st.info("Files already processed in this session.")
        return st.session_state.get('retriever_data', {}).get('documents', [])
    
    all_chunks = []
    session_id = st.session_state.session_id
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract all text first
    all_text = []
    file_info = []
    
    st.info(f"Processing {len(uploaded_files)} file(s): {', '.join([f.name for f in uploaded_files])}")
    
    for file in uploaded_files:
        try:
            # Reset file pointer to beginning
            file.seek(0)
            file_content = io.BytesIO(file.read())
            file_content.name = file.name

            elements = partition(file=file_content)
            full_text = "\n\n".join([el.text for el in elements if hasattr(el, 'text') and el.text])

            if not full_text.strip():
                st.warning(f"No text content found in {file.name}")
                continue

            all_text.append(full_text)
            file_info.append({
                'name': file.name,
                'text': full_text,
                'char_count': len(full_text),
                'token_count': estimate_tokens(full_text)
            })

        except Exception as e:
            st.error(f"Error processing file {file.name}: {e}")
            continue
    
    if not all_text:
        return []
    
    # Combine all text
    combined_text = "\n\n".join(all_text)
    total_tokens = estimate_tokens(combined_text)
    
    # Add metadata header
    metadata_header = f"""DOCUMENT PROCESSING INFORMATION
{'='*80}
Processing Timestamp: {timestamp}
Session ID: {session_id}
Files Processed: {', '.join([info['name'] for info in file_info])}
Total Files: {len(file_info)}
Total Tokens: {total_tokens:,}
Token Limit (Chunking): {token_limit:,}
LLM Token Limit: {llm_token_limit:,}
Chunk Size: {chunk_size} characters
Chunk Overlap: {chunk_overlap} characters
File Signature: {file_signature}
{'='*80}

"""
    
    # Generate filename for this batch
    filename = generate_filename(uploaded_files, is_chunked=(total_tokens > token_limit))
    
    # ENHANCED LOGIC: Consider both individual file limits AND LLM limits
    needs_chunking = False
    chunking_reason = ""
    
    # Check if any individual file exceeds token_limit
    large_files = [info for info in file_info if info['token_count'] > token_limit]
    if large_files:
        needs_chunking = True
        chunking_reason = f"Individual file(s) exceed limit: {[f['name'] for f in large_files]}"
    
    # Check if combined total exceeds LLM_TOKEN_LIMIT (even if individuals are small)
    elif total_tokens > llm_token_limit:
        needs_chunking = True
        chunking_reason = f"Combined total ({total_tokens:,}) exceeds LLM limit ({llm_token_limit:,})"
    
    # Check if combined total exceeds TOKEN_LIMIT (original logic)
    elif total_tokens > token_limit:
        needs_chunking = True
        chunking_reason = f"Combined total ({total_tokens:,}) exceeds chunking limit ({token_limit:,})"
    
    if not needs_chunking:
        # Save as single file without chunking
        st.success(f"📊 Total tokens: {total_tokens:,} - No chunking needed")
        
        content = metadata_header
        for info in file_info:
            content += f"\n{'#'*60}\n"
            content += f"FILE: {info['name']}\n"
            content += f"Characters: {info['char_count']:,}\n"
            content += f"Estimated Tokens: {info['token_count']:,}\n"
            content += f"{'#'*60}\n\n"
            content += info['text']
            content += "\n\n"
        
        saved_path = save_to_file(content, filename)
        if saved_path:
            st.success(f"✅ Text extracted and saved to: {filename}")
        
        # Create single document for retrieval
        from langchain.schema import Document
        doc = Document(
            page_content=combined_text,
            metadata={
                'source': ', '.join([info['name'] for info in file_info]),
                'session_id': session_id,
                'processing_time': datetime.now().isoformat(),
                'is_chunked': False,
                'token_count': total_tokens,
                'file_signature': file_signature
            }
        )
        all_chunks = [doc]
        
    else:
        # Chunking needed
        st.warning(f"⚠️ Chunking required: {chunking_reason}")
        st.info(f"📏 Using chunk size: {chunk_size} chars, overlap: {chunk_overlap} chars")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "],
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks_content = [metadata_header]
        chunks_content.append(f"CHUNKING APPLIED - {chunking_reason}\n")
        chunks_content.append(f"Chunk Size: {chunk_size} characters\n")
        chunks_content.append(f"Chunk Overlap: {chunk_overlap} characters\n")
        
        for info in file_info:
            chunks = text_splitter.create_documents([info['text']])
            
            chunks_content.append(f"\n{'#'*60}")
            chunks_content.append(f"FILE: {info['name']}")
            chunks_content.append(f"Original Characters: {info['char_count']:,}")
            chunks_content.append(f"Original Tokens: {info['token_count']:,}")
            chunks_content.append(f"Chunks Generated: {len(chunks)}")
            chunks_content.append(f"{'#'*60}\n")

            for i, chunk in enumerate(chunks):
                chunk.metadata['source'] = info['name']
                chunk.metadata['chunk_id'] = i
                chunk.metadata['session_id'] = session_id
                chunk.metadata['processing_time'] = datetime.now().isoformat()
                chunk.metadata['is_chunked'] = True
                chunk.metadata['file_signature'] = file_signature
                chunk.metadata['chunk_size'] = chunk_size
                chunk.metadata['chunk_overlap'] = chunk_overlap
                
                chunks_content.append(
                    f"Source: {info['name']} | Chunk {i} | Session: {session_id}\n"
                    f"Content: {chunk.page_content}\n{'-'*50}"
                )

            all_chunks.extend(chunks)
        
        chunks_text = "\n\n".join(chunks_content)
        saved_path = save_to_file(chunks_text, filename)
        if saved_path:
            st.success(f"✅ Text chunked and saved to: {filename}")
            st.info(f"📊 Created {len(all_chunks)} chunks from {total_tokens:,} tokens")

    # Store the file signature to prevent reprocessing
    st.session_state.last_processed_files = file_signature
    
    return all_chunks

# --- Semantic Retrieval Logic ---
def retrieve_and_format_chunks(query, retriever_data):
    """
    Enhanced retrieval with better feedback about processing decisions.
    """
    if not retriever_data:
        st.error("Retriever data is not available. Please process documents first.")
        return []

    # Get current limits
    limits = get_token_limits()
    llm_token_limit = limits['llm_token_limit']
    max_context_chars = limits['max_context_chars']
    max_chunks = limits['max_chunks']

    documents = retriever_data["documents"]
    
    # Calculate total tokens of all documents
    total_tokens = 0
    for doc in documents:
        total_tokens += estimate_tokens(doc.page_content)
    
    # Check if documents were pre-chunked
    is_chunked = any(doc.metadata.get('is_chunked', False) for doc in documents)
    
    if is_chunked:
        st.info(f"📊 Using {len(documents)} pre-chunked documents ({total_tokens:,} tokens)")
    else:
        st.info(f"📊 Using {len(documents)} full document(s) ({total_tokens:,} tokens)")
    
    # Check if we need semantic search or can send all data
    if total_tokens <= llm_token_limit:
        st.success(f"✅ Sending all data to LLM (within {llm_token_limit:,} token limit)")
        
        # Return all documents without semantic search
        all_chunks = []
        for doc in documents:
            chunk_text = (
                f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                f"Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}\n"
                f"Session ID: {doc.metadata.get('session_id', 'Unknown')}\n"
                f"Content: {doc.page_content}"
            )
            all_chunks.append(chunk_text)
        
        return all_chunks
    
    else:
        st.warning(f"⚠️ Using semantic search: {total_tokens:,} tokens > {llm_token_limit:,} limit")
        st.info(f"🔍 Will select top {max_chunks} most relevant chunks (max {max_context_chars:,} chars)")
        
        # Load embedder only when needed
        with st.spinner("Loading embedding model for semantic search..."):
            embedder = load_embedder()
        
        # Get embeddings (create if not exist)
        if "embeddings" not in retriever_data:
            with st.spinner("Creating embeddings for semantic search..."):
                texts = [d.page_content for d in documents]
                from sentence_transformers import util
                embeddings = embedder.encode(texts, convert_to_tensor=True)
                retriever_data["embeddings"] = embeddings
        else:
            embeddings = retriever_data["embeddings"]
            from sentence_transformers import util

        # Compute query embedding
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, embeddings)[0]

        # Pick top-k
        top_k = min(max_chunks, len(cosine_scores))
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
                f"Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}\n"
                f"Session ID: {doc.metadata.get('session_id', 'Unknown')}\n"
                f"Similarity Score: {score:.3f}\n\n"
                f"Content: {doc.page_content}"
            )
            if total_chars + len(chunk_text) > max_context_chars:
                break
            selected_chunks.append(chunk_text)
            total_chars += len(chunk_text)

        st.info(f"📋 Selected {len(selected_chunks)} most relevant chunks via semantic search")
        return selected_chunks

# --- Gemini Model Interaction ---
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
    st.set_page_config(page_title="LLM Document Processing System", layout="wide")
    st.title("⚙️ LLM Document Processing System")
    
    # Initialize session and cleanup
    initialize_session()
    
    # Display session info
    st.sidebar.info(f"Session ID: {st.session_state.session_id}")

    if not client:
        st.error("Google API Key missing or invalid. Configure via .env or Streamlit secrets.")
        return

    # Sidebar: Debug Controls
    with st.sidebar:
        st.header("🔧 Debug Controls")
        with st.expander("Token Limits & Chunk Configuration", expanded=False):
            st.subheader("Processing Limits")
            
            # Chunking Token Limit
            new_token_limit = st.number_input(
                "Chunking Token Limit",
                min_value=10000,
                max_value=2000000,
                value=st.session_state.get('token_limit', DEFAULT_TOKEN_LIMIT),
                step=10000,
                help="Documents exceeding this limit will be chunked during processing"
            )
            
            # LLM Token Limit
            new_llm_token_limit = st.number_input(
                "LLM Token Limit",
                min_value=50000,
                max_value=2000000,
                value=st.session_state.get('llm_token_limit', DEFAULT_LLM_TOKEN_LIMIT),
                step=10000,
                help="If total tokens exceed this, semantic search will be used instead of sending all data"
            )
            
            st.subheader("Chunking Settings")
            
            # Chunk Size
            new_chunk_size = st.number_input(
                "Chunk Size (characters)",
                min_value=500,
                max_value=5000,
                value=st.session_state.get('chunk_size', DEFAULT_CHUNK_SIZE),
                step=100,
                help="Size of each text chunk when documents are split. Larger chunks = more context per chunk but fewer chunks."
            )
            
            # Chunk Overlap
            new_chunk_overlap = st.number_input(
                "Chunk Overlap (characters)",
                min_value=0,
                max_value=1000,
                value=st.session_state.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP),
                step=50,
                help="Character overlap between consecutive chunks. Helps maintain context across chunk boundaries."
            )
            
            st.subheader("Semantic Search Settings")
            
            # Max Context Characters
            new_max_context_chars = st.number_input(
                "Max Context Characters",
                min_value=10000,
                max_value=200000,
                value=st.session_state.get('max_context_chars', DEFAULT_MAX_CONTEXT_CHARS),
                step=5000,
                help="Maximum characters to send to LLM when using semantic search. Controls final context size."
            )
            
            # Max Chunks
            new_max_chunks = st.number_input(
                "Max Chunks for Semantic Search",
                min_value=5,
                max_value=100,
                value=st.session_state.get('max_chunks', DEFAULT_MAX_CHUNKS),
                step=5,
                help="Maximum number of chunks to consider when using semantic search. Higher = more comprehensive but slower search."
            )
            
            # Update session state if values changed
            if (new_token_limit != st.session_state.get('token_limit') or
                new_llm_token_limit != st.session_state.get('llm_token_limit') or
                new_max_context_chars != st.session_state.get('max_context_chars') or
                new_max_chunks != st.session_state.get('max_chunks') or
                new_chunk_size != st.session_state.get('chunk_size') or
                new_chunk_overlap != st.session_state.get('chunk_overlap')):
                
                st.session_state.token_limit = new_token_limit
                st.session_state.llm_token_limit = new_llm_token_limit
                st.session_state.max_context_chars = new_max_context_chars
                st.session_state.max_chunks = new_max_chunks
                st.session_state.chunk_size = new_chunk_size
                st.session_state.chunk_overlap = new_chunk_overlap
                
                # Clear processed files to force reprocessing with new limits
                st.session_state.last_processed_files = None
                if 'retriever_data' in st.session_state:
                    del st.session_state.retriever_data
                
                st.info("🔄 Settings updated! Please re-upload files to apply new configuration.")
            
            # Display current settings
            st.info(f"""
            **Current Settings:**
            - Chunking Limit: {st.session_state.get('token_limit', DEFAULT_TOKEN_LIMIT):,} tokens
            - LLM Limit: {st.session_state.get('llm_token_limit', DEFAULT_LLM_TOKEN_LIMIT):,} tokens
            - Chunk Size: {st.session_state.get('chunk_size', DEFAULT_CHUNK_SIZE)} chars
            - Chunk Overlap: {st.session_state.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP)} chars
            - Max Context: {st.session_state.get('max_context_chars', DEFAULT_MAX_CONTEXT_CHARS):,} chars
            - Max Chunks: {st.session_state.get('max_chunks', DEFAULT_MAX_CHUNKS)}
            """)

        st.header("📤 Upload Documents")
        
        uploaded = st.file_uploader(
            "Choose files", accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'md', 'pptx', 'eml'],
            key="file_uploader"
        )

        if uploaded:
            # Create file signature for this upload batch
            file_signature = create_file_signature(uploaded)
            
            # Check if this is a new upload batch
            if st.session_state.get('last_processed_files') != file_signature:
                with st.spinner("Processing and extracting text..."):
                    docs = process_documents(uploaded)
                    if docs:  # Only update if processing was successful
                        st.session_state.retriever_data = {
                            "documents": docs
                            # Note: embeddings created only when needed
                        }
            else:
                st.info("✅ Files already processed in this session")
                # Show what files are currently loaded
                if st.session_state.get("retriever_data"):
                    docs = st.session_state.retriever_data["documents"]
                    st.success(f"📁 {len(docs)} document(s) currently loaded")

        if st.session_state.get("retriever_data"):
            docs_count = len(st.session_state.retriever_data["documents"])
            st.sidebar.success(f"📁 {docs_count} document(s) ready for queries")

        # File Management Section
        st.header("📁 File Management")
        user_files = get_user_files()
        
        if user_files:
            st.subheader("Saved Files")
            for filename in user_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(filename)
                    # Get and display file statistics
                    char_count, token_count = get_file_stats(filename)
                    if char_count > 0:
                        # Format numbers with commas for readability
                        st.caption(f"📊 {char_count:,} characters • ~{token_count:,} tokens")
                    else:
                        st.caption("📊 Unable to read file stats")
                with col2:
                    if st.button("🗑️", key=f"delete_{filename}", help=f"Delete {filename}"):
                        if delete_file(filename):
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {filename}")
        else:
            st.info("No saved files")

    # Main: query
    st.header("Enter Your Query")
    query = st.text_input("What would you like to know about your documents?", key="query_input")

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