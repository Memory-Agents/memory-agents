# LongMemEval Execution Guide (Oracle Dataset - 500 Questions)

## 1. Environment Setup

See the toplevel README.md of the repository.

## 2. Answer Generation

**Using BaselineAgent (Memory-Augmented Agent):**

```bash
# Run from workspace root
cd /memory_agents/longmemeval
uv run answerGeneration.py
```

**Features:**
- Uses BaselineAgent with in-memory conversation history
- Processes conversation history sequentially to build agent memory
- If interrupted, re-running will skip already processed questions and continue
- Output: `my_predictions.jsonl` file in the current directory

## 3. Evaluation

**Cost:** Approximately $0.4 (using GPT-4o, oracle dataset 500 questions)

```bash
cd src/evaluation
uv run evaluate_qa.py gpt-4o ../../my_predictions.jsonl ../../data/longmemeval_oracle.json
```

**Output Files:**
- `../../my_predictions.jsonl.eval-results-gpt-4o`: Detailed evaluation results for each question
- `../../my_predictions.jsonl.Results_Summary.txt`: Accuracy summary

## 4. Results

Check the generated summary file after evaluation:
```bash
cat ../../my_predictions.jsonl.Results_Summary.txt
```

Example output:
```
Accuracy: 0.8542
	single-session-user: 0.9123 (150)
	multi-session: 0.8234 (120)
	temporal-reasoning: 0.7891 (100)
	knowledge-update: 0.8456 (130)

Saved to ../../my_predictions.jsonl.eval-results-gpt-4o
```