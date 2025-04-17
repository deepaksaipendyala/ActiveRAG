import streamlit as st
import os
import tempfile
import time
from collections import Counter
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.node_parser import SimpleNodeParser
import pypdf
from llama_index.core import Settings
import sys
sys.path.append('.')
from src.generate import create_agent_group, create_plan
from src.prompt import Prompt
from src.metrics import ras_score
import json

# ---------- Confidence Scoring Utilities ----------
def length_confidence(answer):
    return max(0.1, 1.0 - len(answer.split()) / 150)

def agreement_confidence(agent_outputs):
    counts = Counter(agent_outputs.values())
    top_count = counts.most_common(1)[0][1]
    return top_count / len(agent_outputs)

def self_consistency(agent, input_dict, n=3):
    outputs = []
    for _ in range(n):
        agent.message = []
        agent.padding_template(input_dict)
        output = agent.send_message()["choices"][0]["message"]["content"]
        if "Answer:" in output:
            output = output.split("Answer:")[-1].strip()
        outputs.append(output)
    counts = Counter(outputs)
    top_count = counts.most_common(1)[0][1]
    return top_count / n

def hybrid_confidence(agent_name, agent, input_dict, agent_outputs):
    try:
        answer = agent.get_final_answer()
    except:
        return 0.0
    len_conf = length_confidence(answer)
    vote_conf = agreement_confidence(agent_outputs)
    try:
        self_consist_conf = self_consistency(agent, input_dict, n=3)
    except:
        self_consist_conf = 0.0
    final_score = (0.2 * len_conf + 0.4 * vote_conf + 0.4 * self_consist_conf)
    return round(final_score, 4)

# ---------- Streamlit UI Setup ----------
st.set_page_config(page_title="ActiveRAG Document Q&A", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None
if "file_path" not in st.session_state:
    st.session_state.file_path = None
if "answers" not in st.session_state:
    st.session_state.answers = {}

def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf = pypdf.PdfReader(f)
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text

st.title("ActiveRAG PDF Q&A System")

with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    st.header("Model Settings")
    model_selection = st.selectbox("Select LLM Model", ["llama3.2", "llama3", "llama2"], index=0)
    embedding_model = st.selectbox("Select Embedding Model", ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"], index=0)
    chunk_size = st.slider("Chunk Size", min_value=256, max_value=1024, value=512, step=64)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=256, value=50, step=10)
    st.header("Retrieval Settings")
    top_k = st.slider("Number of chunks to retrieve (top_k)", min_value=1, max_value=10, value=5, step=1)
    process_button = st.button("Process Document")

if uploaded_file and process_button:
    with st.spinner("Processing the document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            st.session_state.file_path = temp_file.name

        text = extract_text_from_pdf(st.session_state.file_path)
        embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        llm = Ollama(model=model_selection, request_timeout=120.0)
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = node_parser

        documents = [Document(text=text)]
        st.session_state.index = VectorStoreIndex.from_documents(documents)
        st.success("Document processed successfully!")

st.header("Chat with your document")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_input := st.chat_input("Ask a question about your document"):
    if st.session_state.index is None:
        st.error("Please upload and process a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                query_engine = st.session_state.index.as_query_engine(similarity_top_k=top_k)
                response = query_engine.query(user_input)
                st.write(f"✅ Retrieved {len(response.source_nodes)} relevant chunks in {round(time.time() - start_time, 2)}s")

                passages = [node.text for node in response.source_nodes]
                if not passages:
                    st.error("⚠️ No relevant content found. Try uploading a more detailed document.")
                    st.stop()

                input_dict = {"question": user_input, "passages": passages, "__answers__": [""]}
                group = create_agent_group(Prompt())
                plan = create_plan(group, init_input=input_dict)
                st.write("🛠️ Plan created. Starting execution...")

                execution_start = time.time()
                plan.excute()
                st.write(f"✅ Plan executed in {round(time.time() - execution_start, 2)}s")

                cognition_output = plan.agents.agent_dic['cognition'].get_output()
                final_answer = cognition_output
                if "Answer:" in final_answer:
                    final_answer = final_answer.split("Answer:")[-1].strip()

                message_placeholder = st.empty()
                full_response = ""
                for chunk in final_answer.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.02)
                message_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

                with st.expander("📄 View Source Documents"):
                    for i, node in enumerate(response.source_nodes):
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(node.text)
                        st.markdown("---")

                agent_outputs = {}
                agent_confidences = {}
                for name, agent in plan.agents.agent_dic.items():
                    try:
                        agent_outputs[name] = agent.get_final_answer()
                    except:
                        agent_outputs[name] = "N/A"

                if len(set(agent_outputs.values())) > 2:
                    st.warning("⚠️ Agents produced significantly different answers. Consider reviewing manually.")

                for name, agent in plan.agents.agent_dic.items():
                    score = hybrid_confidence(name, agent, input_dict, agent_outputs)
                    agent_confidences[name] = score

                st.subheader("📊 Annotator Confidence Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Agent Confidences")
                    for name, score in agent_confidences.items():
                        icon = "🟢" if score > 0.7 else ("🟡" if score > 0.4 else "🔴")
                        st.markdown(f"**{name.capitalize()}**: {score} {icon}")
                with col2:
                    st.markdown("### Agent Answers")
                    for name, output in agent_outputs.items():
                        st.markdown(f"**{name.capitalize()}**: {output}")

                with st.expander("🧠 Agent Agreement (RAS Score)"):
                    st.markdown("Higher RAS means more semantic similarity between agents.")
                    agent_names = list(agent_outputs.keys())
                    embed_model = Settings.embed_model  # ✅ fix the NameError
                    for i in range(len(agent_names)):
                        for j in range(i + 1, len(agent_names)):
                            name1 = agent_names[i]
                            name2 = agent_names[j]
                            ras = ras_score(
                                {"query": user_input, "references": agent_outputs[name1]},
                                {"query": user_input, "references": agent_outputs[name2]},
                                embed_model
                            )
                            st.markdown(f"**{name1.capitalize()} ↔ {name2.capitalize()}**: `{round(ras, 4)}`")
