# ActiveRAG: A Continuous Learning Framework for LLMs

Author: Deepak Sai Pendyala  
Course: Efficient Deep Learning (CSC 591/791)  
Advisor: Prof. Dongkuan Xu  
Repository: https://github.com/deepaksaipendyala/ActiveRAG

## Overview

ActiveRAG is a fully local, multi-agent Retrieval-Augmented Generation (RAG) system that leverages active learning, agent consensus, and hybrid confidence scoring to reduce hallucinations in LLM-based question answering, without requiring LLM fine-tuning. It operates on a quantized LLaMA 3.2 model served locally using Ollama and features an interactive Streamlit frontend for querying uploaded PDF documents.

This system achieves a 64% reduction in hallucinations and 30% lower latency compared to a traditional single-agent RAG baseline, with no dependency on paid APIs.

## Key Features

- Multi-agent reasoning framework with five specialized agents: Cot, Anchoring, Associate, Logician, and Cognition
- Hybrid confidence scoring using length penalty, agent voting agreement, and self-consistency check
- Retrieval-Augmented Similarity (RAS) for semantic agreement scoring
- Fully self-hosted architecture with no external API calls
- Streamlit interface for real-time interaction with uploaded PDFs
- Active learning loop for continual improvement through feedback

## Quickstart
### Manual Setup

Requirements:
- Python 3.10+
- Ollama with llama3:8b-instruct-q4_K_M
- FAISS, Streamlit, LlamaIndex, HuggingFace Embeddings, PyPDF, and other dependencies

```pip install -r requirements.txt```


## System Architecture

The pipeline is divided into four modular layers:

1. **Ingestion Layer**  
   Uploaded PDFs are chunked with 512-token windows and 128-token overlap. Each chunk is embedded using ModernBERT and stored in a FAISS index.

2. **Retrieval Layer**  
   Queries trigger a top-k Maximum Inner Product search (typically k = 5). The top passages are used for generation.

3. **Multi-Agent Reasoning Layer**  
   Five agents run in parallel threads, each applying a distinct reasoning strategy. Their answers are consolidated using majority voting and RAS scoring.

4. **Confidence & Feedback Layer**  
   The hybrid confidence score determines if the output should be trusted, flagged, or reviewed. Low-confidence answers are queued for manual annotation and future fine-tuning.

## Agent Roles

- **Cot**: Performs step-by-step chain-of-thought reasoning.
- **Anchoring**: Fills missing background details using local corpus references.
- **Associate**: Aligns the question with cached knowledge and analogies.
- **Logician**: Reformulates answers as logical premises and conclusions.
- **Cognition**: Acts as a meta-critic to verify consistency across agents.

## Confidence Scoring

The hybrid confidence score is computed as follows:

- **Length Penalty**: Penalizes excessively long answers.
- **Voting Agreement**: Measures Jaccard similarity across agent outputs.
- **Self-Consistency**: Reruns Cognition with a new seed and compares output stability.

Scores are used to display trust levels (green, yellow, red) and manage the feedback loop.

## Evaluation Results

| Metric                    | Baseline (Single Agent) | ActiveRAG (Multi-Agent) |
|---------------------------|--------------------------|---------------------------|
| Hallucination Rate        | 27.5%                    | 9.8%                      |
| Mean RAS (Correct Answers)| 0.52                     | 0.87                      |
| End-to-End Latency        | 9.8 seconds              | 6.7 seconds               |
| VRAM Usage                | 5.9 GB                   | 6.1 GB                    |

## Future Work

- Direct Preference Optimization (DPO) using logged preference pairs
- Inline feedback widgets in the UI for immediate annotation
- Lightweight deployment on edge devices (e.g., Raspberry Pi, Snapdragon)
- Learned confidence scoring models to replace heuristics
- Expansion to public benchmarks such as arXiv CS papers
- Development of an explainability dashboard for agent-level insights

## References

1. Xu, Z. et al. (2024). ActiveRAG: Autonomously Knowledge Assimilation and Accommodation through Retrieval-Augmented Agents. ACL 2024. arXiv:2402.13547
2. Geng, X. et al. (2025). AL4RAG: Enhancing RAG with Active Learning on Conversation Records. EMNLP 2025. arXiv:2502.09073
3. Asai, A. et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Reason with Self-Refinement. arXiv:2306.11644
4. Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020
5. Guu, K. et al. (2020). REALM: Retrieval-Augmented Language Model Pretraining. ICML 2020

## Citation

@misc{ActiveRAG2025, author = {Deepak Sai Pendyala}, title = {ActiveRAG: A Continuous Learning Framework for LLMs}, year = {2025}, note = {Efficient Deep Learning — CSC 591/791, North Carolina State University}, url = {https://github.com/deepaksaipendyala/ActiveRAG} }


## License

This project is intended for academic research and educational use. Please contact the author for inquiries regarding commercial or derivative work.


