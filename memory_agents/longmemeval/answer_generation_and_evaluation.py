"""LongMemEval Answer Generation and Evaluation Module.

This module provides functionality to generate answers for the LongMemEval dataset
using various memory agents and evaluate their performance. It supports different
difficulty levels (easy, medium, hard) and agent types (baseline, graphiti, etc.).

The module handles dataset downloading, answer generation with resumption support,
and automated evaluation using the official LongMemEval evaluation scripts.

Example:
    Basic usage with baseline agent:

    >>> from memory_agents.core.agents.baseline import BaselineAgent
    >>> agent = BaselineAgent()
    >>> evaluate("easy", agent)

    Using a subset of questions:

    >>> subset = ["q1", "q2", "q3"]
    >>> evaluate("medium", agent, subset=subset)

"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
import requests
from dotenv import load_dotenv

workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from memory_agents.core.agents.baseline_vdb import BaselineVDBAgent
from memory_agents.core.run_agent import run_agent_messages
from memory_agents.config import (
    LONGMEMEVAL_DIFFICIULTY_LEVEL,
    LONGMEMEVAL_URL_MAP,
)

load_dotenv()


async def generate_answers_with_agent(
    agent: Any,
    dataset_path: str = "data/longmemeval_oracle.json",
    output_path: str = "my_predictions.jsonl",
    subset: list[str] = [],
):
    """
    Generate answers for the LongMemEval dataset using the provided agent.

    Args:
        agent: Agent object (e.g., BaselineAgent or GraphitiAgent)
        dataset_path: Path to the input dataset
        output_path: Path to the output predictions file
        subset: List of question IDs to process
        If subset is empty, all questions will be processed.
    """

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)

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
            if item["question_id"] in processed_ids:
                continue

            thread_id = str(item["question_id"])
            if subset and thread_id not in subset:
                print(f"Skipping question ID: {thread_id} (not in subset)")
                continue
            print(f"Using thread ID: {thread_id}")

            print(
                f"[{idx}/{total}] Processing question ID: {item['question_id']}...",
                end=" ",
                flush=True,
            )

            await agent.clear_agent_memory()
            messages = []
            for date, session in zip(item["haystack_dates"], item["haystack_sessions"]):
                messages.append({"role": "system", "content": f"Date: {date}"})
                for turn in session:
                    messages.append({"role": turn["role"], "content": turn["content"]})

            messages.append({"role": "user", "content": item["question"]})

            hypothesis = await run_agent_messages(
                agent.agent,
                messages,
                thread_id=thread_id,
            )
            out.write(
                json.dumps(
                    {
                        "question_id": item["question_id"],
                        "hypothesis": hypothesis,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()

            print("Done ✓")

    print(f"\n✅ All predictions completed! Results saved to: {output_path}")


def _getDatasetPath(difficulty: str) -> str:
    """Get the dataset file path for a given difficulty level.

    This internal function maps difficulty levels to their corresponding
    dataset file names in the data directory.

    Args:
        difficulty (str): The difficulty level. Must be one of:
            'easy', 'medium', or 'hard'.

    Returns:
        str: The relative path to the dataset file.

    Raises:
        ValueError: If the difficulty level is not one of the supported values.

    Example:
        >>> _getDatasetPath("easy")
        'data/longmemeval_oracle.json'
        >>> _getDatasetPath("medium")
        'data/longmemeval_s_cleaned.json'
    """
    base_path = "data/"
    if difficulty == "easy":
        return os.path.join(base_path, "longmemeval_oracle.json")
    elif difficulty == "medium":
        return os.path.join(base_path, "longmemeval_s_cleaned.json")
    elif difficulty == "hard":
        return os.path.join(base_path, "longmemeval_m_cleaned.json")
    else:
        raise ValueError(
            "Invalid difficulty level. Choose from 'easy', 'medium', or 'hard'."
        )


def getDatasetPathWithCheck(difficulty: str) -> str:
    """Get dataset path with automatic download if file doesn't exist.

    This function checks if the dataset file exists locally. If not, it downloads
    the dataset from the configured URL for the specified difficulty level.

    Args:
        difficulty (str): The difficulty level. Must be one of:
            'easy', 'medium', or 'hard'.

    Returns:
        str: The path to the dataset file (existing or newly downloaded).

    Raises:
        ValueError: If the difficulty level is invalid.
        RuntimeError: If the dataset download fails.
        FileNotFoundError: If the dataset file cannot be created after download.

    Example:
        >>> path = getDatasetPathWithCheck("easy")
        >>> print(f"Dataset available at: {path}")
    """
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
            with open(dataset_path, "w", encoding="utf-8") as f:
                f.write(response.text)
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Failed to download dataset file: {dataset_path}")
    return dataset_path


def evaluate(difficulty, agent, no_generation: bool = False, subset: list[str] = []):
    dataset_path = getDatasetPathWithCheck(difficulty=difficulty)
    output_file_map = {
        "easy": "my_predictions_oracle.jsonl",
        "medium": "my_predictions_s_cleaned.jsonl",
        "hard": "my_predictions_m_cleaned.jsonl",
    }
    output_path = output_file_map[difficulty]
    if not no_generation:
        asyncio.run(
            generate_answers_with_agent(
                agent,
                dataset_path=dataset_path,
                output_path=output_path,
                subset=subset,
            )
        )

    eval_dir = os.path.join(os.path.dirname(__file__), "src/evaluation")
    gold_file_map = {
        "easy": "../../data/longmemeval_oracle.json",
        "medium": "../../data/longmemeval_s_cleaned.json",
        "hard": "../../data/longmemeval_m_cleaned.json",
    }
    gold_file = gold_file_map[difficulty]
    model_name = "gpt-4o"
    print(f"\nRunning evaluation for {difficulty} set...")
    try:
        subprocess.run(
            [
                sys.executable,
                "evaluate_qa.py",
                model_name,
                f"../../{output_path}",
                gold_file,
            ],
            cwd=eval_dir,
            check=True,
        )
    except Exception as e:
        print(f"Evaluation failed: {e}")


if __name__ == "__main__":
    """Command-line interface for LongMemEval evaluation.
    
    This script can be run directly from the command line to evaluate different
    agents on the LongMemEval dataset. It supports various agent types and
    configuration options.
    
    Usage:
        python answer_generation_and_evaluation.py --agent baseline
        python answer_generation_and_evaluation.py --agent graphiti --no_generation
        python answer_generation_and_evaluation.py --agent baseline_vdb --subset_path custom_subset.txt
    
    Args:
        --agent: Agent type to use (baseline, graphiti, graphiti_vdb, baseline_vdb)
        --no_generation: Skip answer generation, only run evaluation
        --subset_path: Path to file containing question IDs to process
    """
    import argparse
    from memory_agents.core.agents.baseline import BaselineAgent
    from memory_agents.core.agents.graphiti import GraphitiAgent
    from memory_agents.core.agents.graphiti_vdb import GraphitiVDBAgent

    parser = argparse.ArgumentParser(
        description="Evaluate memory agents on LongMemEval dataset"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="baseline",
        choices=["baseline", "graphiti", "graphiti_vdb", "baseline_vdb"],
        help="Type of agent to use for evaluation",
    )
    parser.add_argument(
        "--no_generation",
        action="store_true",
        default=False,
        help="Skip answer generation and only run evaluation on existing predictions",
    )
    parser.add_argument(
        "--subset_path",
        type=str,
        default="subset.txt",
        help="Path to file containing question IDs to process (one per line)",
    )
    args = parser.parse_args()

    difficulty = LONGMEMEVAL_DIFFICIULTY_LEVEL
    if args.agent == "baseline":
        agent = BaselineAgent()
    elif args.agent == "graphiti":
        agent = asyncio.run(GraphitiAgent().create())
    elif args.agent == "graphiti_vdb":
        agent = asyncio.run(GraphitiVDBAgent().create())
    elif args.agent == "baseline_vdb":
        agent = BaselineVDBAgent()
    else:
        raise ValueError(f"Invalid agent: {args.agent}")
    subset = []
    if os.path.exists(args.subset_path):
        with open(args.subset_path, "r", encoding="utf-8") as f:
            subset = [line.strip() for line in f]
    evaluate(difficulty, agent, no_generation=args.no_generation, subset=subset)
