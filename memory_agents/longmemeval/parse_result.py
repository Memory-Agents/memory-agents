"""LongMemEval Result Parser Module.

This module parses evaluation results from LongMemEval to extract failed questions
and create a subset file for re-evaluation. It reads the evaluation results file
and outputs question IDs that were incorrectly answered.

Example:
    Basic usage:

    >>> python parse_result.py

    This will read 'my_predictions_oracle.jsonl.eval-results-gpt-4o' and create
    'subset.txt' containing IDs of incorrectly answered questions.

    Programmatic usage:

    >>> parse_evaluation_results("results.jsonl", "failed_questions.txt")
    >>> extract_failed_question_ids("my_predictions_oracle.jsonl.eval-results-gpt-4o")

"""

import json
from typing import List


def parse_evaluation_results(
    input_file: str = "my_predictions_oracle.jsonl.eval-results-gpt-4o",
    output_file: str = "subset.txt",
) -> None:
    """Parse evaluation results and extract failed question IDs.

    This function reads a LongMemEval evaluation results file and extracts
    the IDs of questions that were incorrectly answered (autoeval_label is False).
    The failed question IDs are written to the output file, one per line.

    Args:
        input_file (str, optional): Path to the evaluation results file.
            Defaults to "my_predictions_oracle.jsonl.eval-results-gpt-4o".
        output_file (str, optional): Path to write the failed question IDs.
            Defaults to "subset.txt".

    Raises:
        FileNotFoundError: If the input file does not exist.
        json.JSONDecodeError: If the input file contains invalid JSON.
        IOError: If there are issues reading or writing files.

    Example:
        >>> parse_evaluation_results("results.jsonl", "failed.txt")
    """
    try:
        with open(input_file, "r", encoding="utf-8") as result:
            with open(output_file, "w", encoding="utf-8") as output:
                for line in result:
                    data = line.strip()
                    if data:
                        try:
                            parsed_data = json.loads(data)
                            if (
                                parsed_data.get("autoeval_label", {}).get("label")
                                is False
                            ):
                                question_id = parsed_data.get("question_id")
                                if question_id:
                                    output.write(f"{question_id}\n")
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping malformed JSON line: {e}")
                            continue

        print(
            f"Successfully parsed {input_file} and wrote failed question IDs to {output_file}"
        )

    except (FileNotFoundError, IOError) as e:
        print(f"Error: File issue - {e}")
        raise


def extract_failed_question_ids(input_file: str) -> List[str]:
    """Extract failed question IDs from evaluation results.

    This function reads a LongMemEval evaluation results file and returns a list
    of question IDs that were incorrectly answered.

    Args:
        input_file (str): Path to the evaluation results file.

    Returns:
        List[str]: List of question IDs that failed evaluation.

    Raises:
        FileNotFoundError: If the input file does not exist.
        json.JSONDecodeError: If the input file contains invalid JSON.

    Example:
        >>> failed_ids = extract_failed_question_ids("results.jsonl")
        >>> print(f"Found {len(failed_ids)} failed questions")
    """
    failed_ids = []

    try:
        with open(input_file, "r", encoding="utf-8") as result:
            for line in result:
                data = line.strip()
                if data:
                    try:
                        parsed_data = json.loads(data)
                        if parsed_data.get("autoeval_label", {}).get("label") is False:
                            question_id = parsed_data.get("question_id")
                            if question_id:
                                failed_ids.append(question_id)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON line: {e}")
                        continue

        return failed_ids

    except (FileNotFoundError, IOError) as e:
        print(f"Error: File issue - {e}")
        raise


if __name__ == "__main__":
    """Command-line interface for parsing LongMemEval results.
    
    When run as a script, this module parses the default evaluation results file
    and creates a subset file containing failed question IDs.
    
    Usage:
        python parse_result.py
    """
    try:
        parse_evaluation_results()
    except Exception as e:
        print(f"Result not available, could not parse result: {e}")
