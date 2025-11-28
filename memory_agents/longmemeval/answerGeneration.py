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
# answerGeneration.py -> longmemeval -> memory_agents -> workspace_root
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

# Load environment variables from .env file
load_dotenv()

# Import config
from memory_agents.config import LONGMEMEVAL_URL_MAP

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

    with open(output_path, "a", encoding="utf-8") as out:
        for idx, item in enumerate(dataset, 1):
            # Skip already processed questions
            if item["question_id"] in processed_ids:
                continue

            # Use a different thread_id for each question_id
            thread_id = str(item["question_id"])
            print(f"Using thread ID: {thread_id}")

            print(
                f"[{idx}/{total}] Processing question ID: {item['question_id']}...",
                end=" ",
                flush=True,
            )

            # Build messages list
            messages = []
            for date, session in zip(item["haystack_dates"], item["haystack_sessions"]):
                # Add date information
                messages.append({"role": "system", "content": f"Date: {date}"})
                # Add conversation turns within the session (use role as is)
                for turn in session:
                    messages.append(
                        {"role": turn["role"], "content": turn["content"]}
                    )

            # Add the final question
            messages.append({"role": "user", "content": item["question"]})

            # Pass all messages to the agent at once
            hypothesis = await run_agent_messages(
                agent.agent,
                messages,
                thread_id=thread_id,
            )
            out.write(
                json.dumps(
                    {"question_id": item["question_id"], "hypothesis": hypothesis},
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()  # Write to disk immediately

            print(f"Done ✓")

    print(f"\n✅ All predictions completed! Results saved to: {output_path}")

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
    difficulty = "easy"  # Set difficulty here: "easy", "medium", or "hard"
    agent = BaselineAgent()
    evaluate(difficulty, agent)

    



