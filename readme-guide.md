# Modified ActiveRAG with Llama 3.2 & Ollama

This is a modified implementation of ActiveRAG (Autonomously Knowledge Assimilation and Accommodation through Retrieval-Augmented Agents) that uses Llama 3.2 via Ollama instead of OpenAI API. This implementation aims to reduce costs, address privacy concerns, and provide a local, self-contained RAG system.

## Features

- **Local LLM Integration**: Uses Llama 3.2 through Ollama for all agents
- **Cost Efficient**: No API costs or usage limitations
- **Privacy Focused**: All processing happens locally
- **Multi-Agent Architecture**: Implements the four-agent design (Anchoring, Associate, Logician, Cognition)
- **Streamlit Interface**: Web interface for PDF upload and Q&A
- **Enhanced Prompts**: Optimized prompts for local LLMs
- **ThreadPoolExecutor**: Efficient parallel processing without async complexity

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/activerag-llama
cd activerag-llama
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Install Ollama if you haven't already:

```bash
# For macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# For Windows, download from https://ollama.com/download
```

4. Pull the Llama 3.2 model (or your preferred model):

```bash
ollama pull llama3.2
```

## Usage

### Running the Streamlit Interface

To use the PDF Q&A system with a graphical interface:

```bash
streamlit run streamlit_app.py
```

Then:
1. Upload your PDF document
2. Configure settings if needed
3. Click "Process Document"
4. Start asking questions!

### Running from the Command Line

To process a dataset with the ActiveRAG framework:

```bash
python run_activerag.py --dataset nq --topk 5
```

Options:
- `--dataset`: The dataset name (should match a file in the `sampled/` directory)
- `--topk`: Number of passages to retrieve for each question
- `--model`: Model name (default: llama3.2)
- `--output_dir`: Directory for output files (default: log/activerag)

### Analyzing Results

To build a CSV file with results:

```bash
python -m scripts.build --dataset nq --topk 5
```

To evaluate the results:

```bash
python -m scripts.evaluate --dataset nq --topk 5
```

## Project Structure

- `src/`: Core components
  - `agent.py`: Modified Agent class that works with Ollama
  - `prompt.py`: Enhanced prompts for local LLM
  - `group.py`: Agent coordination with ThreadPoolExecutor
  - `plan.py`: Execution planning
  - `metrics.py`: Evaluation metrics
  - `generate.py`: Agent generation and plan creation
- `scripts/`: Utility scripts
  - `run.py`: Original run script
  - `build.py`: Builds CSV results
  - `evaluate.py`: Evaluates performance
- `streamlit_app.py`: Web interface for PDF Q&A
- `run_activerag.py`: Main entry point with improved logging
- `sampled/`: Sample datasets

## Modifications from Original Implementation

1. Replaced OpenAI API calls with Ollama for local inference
2. Improved prompt engineering for better performance with local LLMs
3. Simplified asynchronous operations using ThreadPoolExecutor
4. Added Streamlit interface for interactive PDF Q&A
5. Enhanced logging and error handling
6. Implemented retries for robustness

## Performance Considerations

- Local LLMs may be slower than API-based solutions
- Performance depends on your hardware (GPU recommended)
- Adjust chunk sizes and retrieval parameters for optimal results
- Consider using smaller models if facing memory constraints

## Troubleshooting

- **Ollama connection issues**: Ensure Ollama is running with `ollama serve`
- **Memory errors**: Reduce batch sizes or use a smaller model
- **Slow performance**: Enable GPU acceleration if available

## Future Improvements

- Implement the AL4RAG active learning techniques
- Add DPO fine-tuning for better accuracy
- Expand to dialogue systems beyond simple Q&A
- Optimize for lower resource settings
