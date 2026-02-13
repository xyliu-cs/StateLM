# StateLM
<p align="center">
  <strong>🤗 <a href="https://huggingface.co/collections/lindsay21/statelm">Models</a></strong> |
  <strong>📝 <a href="https://arxiv.org/abs/2602.12108">Paper</a></strong> |
  <strong>💻 <a href="https://github.com/xyliu-cs/StateLM">Repo</a></strong>
</p>

StateLM is a language model agent equipped with specialized frameworks for diverse tasks such as long-context reasoning, and deep research.
<img src="assets/demo.gif" width="800">

## Overview
<img src="assets/statelm.png" alt="StateLM Overview" width="800px" />

**The self-context management workflow of StateLM.**
Given a query over a long context, StateLM engages in a multi-round, stateful reasoning loop that analyzes the input, builds an index, and iteratively searches, reads, takes notes, and prunes its working context. Messages highlighted in red are replaced with stubs after the deletion operation. The loop terminates once StateLM determines it has gathered sufficient information for the final answer.




## Setup

### 1. Recommended environment (inference)

```bash
conda create -n statelm python=3.12.11 -y
conda activate statelm
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_min.txt 
```

### 2. Elasticsearch

StateLM's `searchEngine` tool requires a running Elasticsearch instance. You can use Docker (recommended) or install manually. Detailed setup guide please refer to: [Elasticsearch Setup Guide](elasticsearch_setup.md).

### 3. vLLM server

[Start a vLLM server](scripts/start_server_statelm_8b.sh) with tool-calling support and create an endpoint config file (e.g. `openai_endpoint_example.json`):

```json
{
  "OPENAI_BASE_URL": "http://localhost:8080/v1",
  "OPENAI_API_KEY": "EMPTY",
  "MODEL_ID": "StateLM-8B"
}
```

## Evaluation

Evaluation uses `scripts/hf_test_runner.py` to run StateLM on HuggingFace-styled datasets (either hosted or local jsonl). Example evaluation scripts are provided in `scripts/`.

### Evaluation on custom dataset
To evaluate on your own dataset, you need to (1) add processing functions and (2) set the configuration and (3) define the grading functions.

**Step 1: Add processing functions**

Create three (or four) functions in `inference/hf_process_fns.py` that map a single dataset item to the fields StateLM expects:

| Function | Purpose | Returns |
|---|---|---|
| `item_to_question` | Extract the **question / query** | `str` |
| `item_to_context` | Extract the **context** (the long document) | `str` |
| `item_to_answer` | Extract the **gold answer** | `str` or `List[str]` |
| `item_to_meta` *(optional)* | Extract any extra metadata to carry through | `Dict[str, Any]` |

For example, for a QA dataset where each item has `"question"`, `"document"`, and `"answer"` fields:

```python
# in inference/hf_process_fns.py

def my_dataset_i2q(item: Dict[str, Any]) -> str:
    return item["question"]

def my_dataset_i2c(item: Dict[str, Any]) -> str:
    return item["document"]

def my_dataset_i2a(item: Dict[str, Any]) -> str:
    return item["answer"]
```

**Step 2: Set the configuration**

Point the runner at your dataset and processing functions. The dataset can be a HuggingFace dataset name or a local `.jsonl` file (use `--dataset_split local` for local files). Typical configurations include:

| Argument | Description |
|---|---|
| `--dataset_name` | HuggingFace dataset ID or path to a local `.jsonl` file |
| `--dataset_split` | Dataset split name (e.g. `test`), or `local` for local `.jsonl` files |
| `--tool_config_path` | e.g.,`statelm_tools.json` |
| `--system_prompt_name` | e.g.,`STATELM_SYSTEM_PROMPT` |
| `--temperature` | Temperature for LLM inference |
| `--max_output_tokens` | Max output tokens for LLM inference |
| `--max_context_exp` | Max context window size in tokens for budget calculation |
| `--max_turns_exp` | Max number of interaction turns |

**Step 3: Define the grading functions**
By default, the output will be saved to a JSONL file in the format of 
```json
{
  "dataset": "...",
  "split": "...",
  "model": "...",
  "sample_id": "...",
  "question": "...",
  "correct_answer": "...",
  "meta_info": {
    "...": "..."
  }
}
```
You need to define the grading functions for this kind of output. Example grading functions are defined in `inference/compute_scores.py`.

## Training
Please refer to our RL implementation based on verl v0.6.0 in `verl/`.

## Citation
```bibtex
@misc{liu2026pensieveparadigmstatefullanguage,
      title={The Pensieve Paradigm: Stateful Language Models Mastering Their Own Context}, 
      author={Xiaoyuan Liu and Tian Liang and Dongyang Ma and Deyu Zhou and Haitao Mi and Pinjia He and Yan Wang},
      year={2026},
      eprint={2602.12108},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.12108}, 
}
```
