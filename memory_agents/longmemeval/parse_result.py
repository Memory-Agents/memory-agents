import json

try:
    result = open("my_predictions_oracle.jsonl.eval-results-gpt-4o", "r")
    output = open("subset.txt", "w")

    for line in result:
        data = line.strip()
        data = json.loads(data)
        if data["autoeval_label"]["label"] == False:
            output.write(data["question_id"] + "\n")

    output.close()
    result.close()
except:
    print("Result not available, could not parse result")
