import argparse
import json
import os
from tqdm import tqdm
import time
import traceback
from datetime import datetime
import sys

# Import from src directory
from src.generate import create_agent_group, create_plan
from src.prompt import Prompt

def main():
    parser = argparse.ArgumentParser(description="Run ActiveRAG with Llama 3.2")
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., nq, triviaqa)')
    parser.add_argument('--topk', type=int, required=True, help='Number of passages to retrieve')
    parser.add_argument('--model', default='llama3.2', help='Model name (default: llama3.2)')
    parser.add_argument('--output_dir', default='log/activerag', help='Output directory for logs')
    args = parser.parse_args()

    dataset = args.dataset
    filename = f'sampled/data_{dataset}_sampled.jsonl'
    topk = args.topk
    model = args.model

    # Create output directory
    directory = f'{args.output_dir}/{dataset}/top{topk}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    print(f"Running ActiveRAG with {model} on {dataset} dataset with top-{topk} passages")
    print(f"Loading data from {filename}")
    print(f"Saving results to {directory}")

    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found")
        return

    # Process each question
    with open(filename, 'r', encoding='utf-8') as file:
        for i, line in tqdm(enumerate(file), desc="Processing questions"):
            try:
                # Load data
                data = json.loads(line)
                question = data['question']
                answers = data['answers']
                
                # Use only top-k passages
                passages = data['passages'][:topk]

                # Prepare input for ActiveRAG
                input_data = {
                    'question': question,
                    'passages': passages,
                    '__answers__': answers,
                }

                # Create and execute plan
                print(f"\nQuestion {i}: {question}")
                start_time = time.time()
                
                plan = create_plan(create_agent_group(Prompt()), init_input=input_data)
                plan.excute()
                
                end_time = time.time()
                print(f"Processing time: {end_time - start_time:.2f} seconds")

                # Save log
                output_file = f'{directory}/{dataset}_idx_{i}.json'
                plan.save_log(output_file)
                print(f"Saved results to {output_file}")

            except KeyboardInterrupt:
                print("\nProcess interrupted by user")
                break
            except Exception as e:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Error at index {i} - {current_time}: {e}")
                traceback.print_exc()
                # Continue to next question instead of breaking
                continue

    print("Processing complete!")

if __name__ == "__main__":
    main()
