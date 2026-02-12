from random import choices
from typing import Dict, Any, Callable, List
import json
import os
from collections import defaultdict
import random

def read_jsonl(fp):
    data = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def write_jsonl(data, fp):
    """Write list of dictionaries to JSONL file."""
    with open(fp, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def nqa_choice_i2q(item: Dict[str, Any]) -> str:
    question = item["Question"]
    options = item["Options"]
    question_str = f"Question: {question}\n"
    for key, val in options.items():
        question_str += f"{key}. {val} \n"
    return question_str

def nqa_choice_i2c(item: Dict[str, Any]) -> str:
    book_id = item['book_id']
    base_dir = 'path/to/your/book/directory'
    book_path = os.path.join(base_dir, f'{book_id}.txt')
    with open(book_path, 'r', encoding='utf-8') as f:
        book_content = f.read()
    return book_content

def nqa_choice_i2a(item: Dict[str, Any]) -> str:
    mcq_answer = item["Gold"]
    return mcq_answer

def infinitebench_longbook_choice_eng_i2q(item: Dict[str, Any]) -> str:
    question = item['input'].strip()
    choices = item['options']
    question_str = f"{question}\n"
    for i, choice in enumerate(choices):
        question_str += f"{chr(65 + i)}. {choice}\n"
    question_str += "Select the best answer from the options above."
    return question_str

def infinitebench_longbook_choice_eng_i2c(item: Dict[str, Any]) -> str:
    return item['context'].strip()

def infinitebench_longbook_choice_eng_i2a(item: Dict[str, Any]) -> str:
    options = item['options']
    answer = item['answer'][0]
    answer_index = options.index(answer)
    return chr(65 + answer_index)


def ruler_niah_i2q(item: Dict[str, Any]) -> str:
    return item['input'].split('\n')[-1]

def ruler_niah_i2c(item: Dict[str, Any]) -> str:
    return '\n'.join(item['input'].split('\n')[1:-1])

def ruler_niah_i2a(item: Dict[str, Any]) -> List[str]:
    return item['outputs']

def ruler_niah_i2meta(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
            "index": item['index'],
            "task_name": item['task_name'],
            "length": item['length'],
            "token_position_answer": item['token_position_answer']
        }

# postprocessing: split output by task name
def ruler_niah_postprocess(output_fp):
    # Read the jsonl file and group by task name on the fly
    task_groups = defaultdict(list)
    with open(output_fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                task_name = item.get('task_name')
                task_groups[task_name].append(item)
    
    # Write separate files for each task
    base_dir = os.path.dirname(output_fp)

    for task_name, items in task_groups.items():
        output_file = os.path.join(base_dir, f"{task_name}.jsonl")
        write_jsonl(items, output_file)
        print(f'Split {len(items)} items to {output_file}.')


def longmemevals_i2q(item: Dict[str, Any]) -> str:
    question_str = item['question'].strip()
    # question_str += "\nAnswer the question based on the attached chat history."
    question_str += "\nAnswer the question by analyzing the attached text."
    return question_str

def longmemevals_i2c(item: Dict[str, Any]) -> str:
    return item['conversation_str'].strip()

def longmemevals_i2a(item: Dict[str, Any]) -> str:
    return item['answer']

def longmemevals_i2meta(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
            "question_id": item['question_id'],
        }

def bc_plus_i2q(item: Dict[str, Any]) -> str:
    return item['query']

def bc_plus_i2c(item: Dict[str, Any]) -> str:
    rng = random.Random(42)   # fixed seed, isolated RNG
    evidence_doc_text = [item['text'] for item in item['evidence_docs']]
    negative_doc_text = [item['text'] for item in item['negative_docs']]
    all_docs_text = evidence_doc_text + negative_doc_text
    rng.shuffle(all_docs_text)
    context = '\n\n'.join(all_docs_text)
    return context

def bc_plus_i2a(item: Dict[str, Any]) -> str:
    return item['answer']