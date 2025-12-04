# LongMemEval Execution Guide (Oracle Dataset - 500 Questions)

## 1. Environment Setup

See the toplevel README.md of the repository.

## 2. Answer Generation

**Using BaselineAgent (Memory-Augmented Agent):**

```bash
cd /memory_agents/longmemeval
uv run answer_generation_and_evaluation.py --agent="graphiti_vdb"
```

## 3. Check the results

**Output Files:**

- `../../my_predictions.jsonl.eval-results-gpt-4o`: Detailed evaluation results for each question
- `../../my_predictions.jsonl.Results_Summary.txt`: Accuracy summary
