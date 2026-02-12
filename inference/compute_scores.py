# ==============================
# Compute scores for MCQ task
# Logic adapted from https://github.com/OpenBMB/InfiniteBench/blob/51d9b37b0f1790ead936df2243abbf7f0420e439/src/compute_scores.py
# ==============================

from pathlib import Path
import json
import re
from collections import Counter

from tqdm import tqdm

# from args import parse_args
import fire, os



def load_json(fname):
    return json.load(open(fname))


def iter_jsonl(fname, cnt=None):
    i = 0
    with open(fname, "r", encoding="utf8") as fin:
        for line in fin:
            if line.strip() == "":  # Skip empty lines
                continue
            if i == cnt:
                break
            if line.strip() == "":  # Skip empty lines
                continue
            yield json.loads(line)
            i += 1

def get_score_one_longbook_choice_eng(pred, label, model_name: str) -> bool:
    # Just use the first letter as the prediction
    pred = pred.strip()
    pattern = r"\b[A-D]\b(?!.*\b[A-D]\b)"

    match = re.search(pattern, pred)
    if match:
        extracted_pred = match.group(0)
        if extracted_pred in label:
            return True
    if pred == "":
        return False
    if pred[0] in "ABCD":
        return pred[0] in label
    if pred in label:
        return True
    # Find a answer prefix
    for c in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    ans_prefixes = [
        "answer is:",
        "answer:",
        "answer is",
        "option is",
    ]
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        # The prediction ends with this prefix
        if len(pred) < idx + len(prefix) + 1:
            return False
        after_prefix = pred[idx + len(prefix) + 1 :]
        for s in label:
            if after_prefix.startswith(s):
                return True
        return False

    # Finally, just find the first occurrence of A, B, C, or D.
    words = pred.split()
    for word in words:
        if word in "ABCD":
            return word in label
    return False


def get_score_one(
    pred: str, label: str, task_name: str, model_name: str
) -> float:
    """
    Computes the score for one prediction.
    Returns one float (zero and one for boolean values).
    """
    NAME_TO_SCORE_GETTER = {
        # Longbook
        "longbook_choice_eng": get_score_one_longbook_choice_eng,
    }
    assert task_name in NAME_TO_SCORE_GETTER, f"Invalid task name: {task_name}"
    score = NAME_TO_SCORE_GETTER[task_name](pred, label, model_name)
    return float(score)


def get_labels(preds: list, label_key: str = None) -> list[str]:
    if label_key is None:
        possible_label_keys = ["ground_truth", "label"]
    else:
        possible_label_keys = [label_key]
    for label_key in possible_label_keys:
        if label_key in preds[0]:
            return [x.get(label_key, "XXXXXXXXXX") for x in preds]
    raise ValueError(f"Cannot find label in {preds[0]}")


def get_preds(preds: list, data_name: str, ans_key: str = None) -> list[str]:
    pred_strings = []
    if ans_key is None:
        possible_pred_keys = ["prediction", "pred"]
    else:
        possible_pred_keys = [ans_key]
    for pred in preds:
        this_pred = "NO PREDICTION"
        for pred_key in possible_pred_keys:
            if pred_key in pred:
                if isinstance(pred[pred_key], list):
                    this_pred = pred[pred_key][0]
                else:
                    this_pred = pred[pred_key]
                break
        else:
            raise ValueError(f"Cannot find prediction in {pred}")
        pred_strings.append(this_pred)
    return pred_strings


def get_score(
    labels: list, preds: list, data_name: str, model_name: str
) -> float:
    """
    Computes the average score for a task.
    """
    assert len(labels) == len(preds)
    scores = []
    for label, pred in tqdm(zip(labels, preds)):
        score = get_score_one(pred, label, data_name, model_name)
        scores.append(score)
    return sum(scores) / len(scores)


QA_TASKS = [
    "longbook_choice_eng",
]

def compute_scores(preds_path: str, task_name: str, results_output: str, model_name: str = 'unknown', label_key: str = None, pred_key: str = None):
    print("Loading prediction results from", preds_path)
    preds = list(iter_jsonl(preds_path))
    labels = get_labels(preds, label_key)
    preds = get_preds(preds, task_name, pred_key)
    acc = get_score(labels, preds, task_name, model_name)
    
    print(f"Model: {model_name}")
    print(f"Task: {task_name} ({len(labels)} examples)")
    print(f"Score: {acc}")
    print("-" * 20)

    os.makedirs(os.path.dirname(results_output), exist_ok=True)
    with open(results_output, 'a') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Results Path: {preds_path}\n")
        f.write(f"Task: {task_name} ({len(labels)} examples)\n")
        f.write(f"Score: {acc}\n")
        f.write("-" * 20 + "\n")
    print("Results written to", results_output)


def evaluate_choice_file(file_path: str, pred_key: str, label_key: str, output_key: str = "correct_list"):
    """
    Evaluate choice-question predictions in a JSONL file.

    Args:
        file_path: path to the JSONL file
        pred_key: key for model predictions
        label_key: key for ground-truth labels
        output_key: key to store correctness list (default "correct_list")
    """
    samples = list(iter_jsonl(file_path))
    results = []

    for sample in samples:
        pred = sample.get(pred_key, "")
        if isinstance(pred, list):
            pred = pred[0]
        label = sample.get(label_key, "")

        # call the existing scoring function
        score = get_score_one(pred, label, "longbook_choice_eng", "model")
        sample[output_key] = [bool(score)]  # wrap as list [True]/[False]
        results.append(sample)

    # overwrite the file with updated samples
    with open(file_path, "w", encoding="utf8") as fout:
        for sample in results:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Updated {len(results)} samples in {file_path} with correctness in '{output_key}'.")

if __name__ == "__main__":
    fire.Fire()