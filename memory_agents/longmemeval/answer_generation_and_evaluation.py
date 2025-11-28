import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
import requests
from dotenv import load_dotenv

# Add the workspace root to Python path for absolute imports
# answer_generation_and_evaluation.py -> longmemeval -> memory_agents -> workspace_root
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

# Load environment variables from .env file
load_dotenv()

# Import config
from memory_agents.config import LONGMEMEVAL_DIFFICIULTY_LEVEL, LONGMEMEVAL_URL_MAP

async def generate_answers_with_agent(
    agent: Any,
    dataset_path: str = "data/longmemeval_oracle.json",
    output_path: str = "my_predictions.jsonl",
):
    """
    Generate answers for the LongMemEval dataset using the provided agent.

    Args:
        agent: Agent object (e.g., BaselineAgent or GraphitiAgent)
        dataset_path: Path to the input dataset
        output_path: Path to the output predictions file
    """
    from memory_agents.core.run_agent import  run_agent_messages

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)

    # Check for already processed question IDs
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        processed_ids.add(data["question_id"])
                    except Exception as e:
                        print(f"Skipping malformed line in output: {e}")
    print(f"⏭️  Already processed: {len(processed_ids)} questions (skipping)")

    remaining = total - len(processed_ids)
    print(f"Starting to process {total} questions... (Remaining: {remaining})\n")


def _getDatasetPath(difficulty: str) -> str:
    base_path = "data/"
    if difficulty == "easy":
        return os.path.join(base_path, "longmemeval_oracle.json")
    elif difficulty == "medium":
        return os.path.join(base_path, "longmemeval_s_cleaned.json")
    elif difficulty == "hard":
        return os.path.join(base_path, "longmemeval_m_cleaned.json")
    else:
        raise ValueError("Invalid difficulty level. Choose from 'easy', 'medium', or 'hard'.")

def getDatasetPathWithCheck(difficulty: str) -> str:
    dataset_path = _getDatasetPath(difficulty)
    if not os.path.exists(dataset_path):
        # Get URL from config
        url = LONGMEMEVAL_URL_MAP.get(difficulty)
        if url is None:
            raise ValueError(f"Invalid difficulty: {difficulty}")
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        print(f"Dataset file not found: {dataset_path}\nDownloading from {url} ...")
        try:
            response = requests.get(url, timeout=300)  # 5 minute timeout
            response.raise_for_status()
            with open(dataset_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Failed to download dataset file: {dataset_path}")
    return dataset_path

def evaluate(difficulty, agent):
    # Map difficulty to dataset and output file
    dataset_path = getDatasetPathWithCheck(difficulty=difficulty)
    output_file_map = {
        "easy": "my_predictions_oracle.jsonl",
        "medium": "my_predictions_s_cleaned.jsonl",
        "hard": "my_predictions_m_cleaned.jsonl",
    }
    output_path = output_file_map[difficulty]
    asyncio.run(generate_answers_with_agent(agent, dataset_path=dataset_path, output_path=output_path))

    # The evaluation script expects to be run from the src/evaluation directory
    eval_dir = os.path.join(os.path.dirname(__file__), "src/evaluation")
    # Map difficulty to gold file
    gold_file_map = {
        "easy": "../../data/longmemeval_oracle.json",
        "medium": "../../data/longmemeval_s_cleaned.json",
        "hard": "../../data/longmemeval_m_cleaned.json",
    }
    gold_file = gold_file_map[difficulty]
    # Model name can be changed as needed
    model_name = "gpt-4o"
    print(f"\nRunning evaluation for {difficulty} set...")
    try:
        subprocess.run([
            sys.executable, "evaluate_qa.py", model_name, f"../../{output_path}", gold_file
        ], cwd=eval_dir, check=True)
    except Exception as e:
        print(f"Evaluation failed: {e}")


if __name__ == "__main__":
    from memory_agents.core.agents.baseline import BaselineAgent
    difficulty = LONGMEMEVAL_DIFFICIULTY_LEVEL  # Set difficulty here: "easy", "medium", or "hard"
    agent = BaselineAgent()
    evaluate(difficulty, agent)

    



