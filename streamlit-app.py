import streamlit as st
import os
import tempfile
import time
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.node_parser import SimpleNodeParser
import pypdf
from llama_index.core import Settings
from sentence_transformers import SentenceTransformer, util
import sys
sys.path.append('.')
from src.generate import create_agent_group, create_plan
from src.prompt import Prompt
import json

# Load the rerank model
rerank_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to rerank passages based on cosine similarity
def rerank_passages(question, passages, topk=3):
    """Reranks retrieved passages based on cosine similarity to the question."""
    inputs = [question] + passages
    embeddings = rerank_model.encode(inputs, convert_to_tensor=True)
    
    question_emb = embeddings[0]
    passage_embs = embeddings[1:]
    
    scores = util.pytorch_cos_sim(question_emb, passage_embs)[0]
    top_indices = scores.argsort(descending=True)[:topk]
    
    return [passages[i] for i in top_indices]

# Set page configuration
st.set_page_config(page_title="ActiveRAG Document Q&A", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "file_path" not in st.session_state:
    st.session_state.file_path = None

if "answers" not in st.session_state:
    st.session_state.answers = {}

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf = pypdf.PdfReader(f)
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text

# Main UI elements
st.title("ActiveRAG PDF Q&A System")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    st.header("Model Settings")
    model_selection = st.selectbox(
        "Select LLM Model",
        ["llama3.2:1b", "llama3.2", "llama3", "llama2"],
        index=0
    )
    
    embedding_model = st.selectbox(
        "Select Embedding Model",
        ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"],
        index=0
    )
    
    chunk_size = st.slider("Chunk Size", min_value=256, max_value=1024, value=512, step=64)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=256, value=50, step=10)
    
    st.header("Retrieval Settings")
    top_k = st.slider("Number of chunks to retrieve (top_k)", min_value=1, max_value=10, value=5, step=1)
    
    process_button = st.button("Process Document")

# Process the uploaded file
if uploaded_file and process_button:
    with st.spinner("Processing the document..."):
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            st.session_state.file_path = temp_file.name
        
        # Extract text from PDF
        text = extract_text_from_pdf(st.session_state.file_path)
        
        # Create embeddings and index
        embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        llm = Ollama(model=model_selection, request_timeout=120.0)
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )


        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = node_parser

        
        documents = [Document(text=text)]
        st.session_state.index = VectorStoreIndex.from_documents(documents)
        
        st.success("Document processed successfully!")

# Display chat interface
st.header("Chat with your document")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
if user_input := st.chat_input("Ask a question about your document"):
    if st.session_state.index is None:
        st.error("Please upload and process a document first.")
    else:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                st.write("🔍 Querying vector index...")
                query_engine = st.session_state.index.as_query_engine(similarity_top_k=top_k)
                response = query_engine.query(user_input)
                st.write(f"✅ Retrieved {len(response.source_nodes)} relevant chunks in {round(time.time() - start_time, 2)}s")

                passages = [node.text for node in response.source_nodes]
                if not passages:
                    st.error("⚠️ No relevant content found. Try uploading a more detailed document.")
                    st.stop()

                st.write("🔄 Reranking passages based on relevance...")
                passages = rerank_passages(user_input, passages, topk=3)
                st.write("🧠 Setting up ActiveRAG agent group...")
                input_dict = {
                    'question': user_input,
                    'passages': passages,
                    '__answers__': [""]
                }

                group = create_agent_group(Prompt())
                plan = create_plan(group, init_input=input_dict)
                st.write("🛠️ Plan created. Starting execution...")

                execution_start = time.time()
                plan.excute()  # ✅ assumes you fixed the typo in your Plan class
                st.write(f"✅ Plan executed in {round(time.time() - execution_start, 2)}s")

                cognition_output = plan.agents.agent_dic['cognition'].get_output()
                st.write("🧾 Raw cognition output:", cognition_output)

                # Extract just the answer part
                final_answer = cognition_output
                if "Answer:" in final_answer:
                    final_answer = final_answer.split("Answer:")[-1].strip()

                # Streaming effect
                message_placeholder = st.empty()
                full_response = ""
                for chunk in final_answer.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.02)
                message_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # Display source documents
                with st.expander("📄 View Source Documents"):
                    for i, node in enumerate(response.source_nodes):
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(node.text)
                        st.markdown("---")
