import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Add the workspace root to Python path for absolute imports
# answerGeneration.py -> longmemeval -> memory_agents -> workspace_root
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

# Load environment variables from .env file
load_dotenv()


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
    from memory_agents.core.run_agent import run_agent_messages

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

            # 각 question_id 별로 다른 thread_id 사용
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
                # 날짜 정보 추가
                messages.append({"role": "system", "content": f"Date: {date}"})
                # 세션 내 대화 추가 (role 그대로 사용)
                for turn in session:
                    messages.append({"role": turn["role"], "content": turn["content"]})

            # 최종 질문 추가
            messages.append({"role": "user", "content": item["question"]})

            # 전체 messages를 한 번에 에이전트에 전달
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


if __name__ == "__main__":
    from memory_agents.core.agents.baseline import BaselineAgent

    baseline_agent = BaselineAgent()
    asyncio.run(generate_answers_with_agent(baseline_agent))
