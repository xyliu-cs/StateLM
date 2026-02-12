#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HF → Agent Runner (openai_orchestrator-based) with simple internal sharding.
"""

import argparse
import importlib
import json
import os
import sys
from typing import Any, Callable, Dict, Optional, Tuple
import importlib.util
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import fire
import multiprocessing as mp

import re
from glob import glob
from datetime import datetime

# -----------------------------
# Orchestrator import shim
# -----------------------------
def _import_statelm(version: str):
    """
    Import the statelm from statelm.py.
    """
    try:
        from StateLM.src.statelm import StateLM
        from StateLM.src.statelm import ExecLogger
    except Exception as e:
        raise ImportError(
            "Could not import StateLM and ExecLogger from statelm.py. "
            f"Original error: {e}"
        )
    return StateLM, ExecLogger

def _load_callable(spec: str) -> Callable:
    """
    Load a callable from 'pkg.module:func' or '/path/to/file.py:func'.
    """
    if ":" not in spec:
        raise ValueError(f"Function spec must be 'module_or_path:func', got: {spec}")
    mod_part, func_name = spec.split(":", 1)

    if mod_part.endswith(".py") or "/" in mod_part or mod_part.startswith("."):
        path = Path(mod_part).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Function file not found: {path}")
        module_name = path.stem + "_dyn"
        spec_obj = importlib.util.spec_from_file_location(module_name, str(path))
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Cannot load module from {path}")
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_part)

    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise AttributeError(f"'{func_name}' not found or not callable in {mod_part}")
    return fn

def read_json(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# Helpers for resume-after-break
# -----------------------------
def _collect_done_ids(result_dir: str, log_dir: str, shard_fp: str) -> set:
    """
    Return a set of sample_ids considered 'done' based on existing artifacts.
    - results_dir: looks for '{sample_id}_final_result_*.json'
    - shard_fp: reads existing jsonl lines (if any) and collects 'sample_id'
    - log_dir: (best-effort) looks for '{sample_id}_inference_result_*.json'
    """
    done = set()

    # 1) Final results: {sample_id}_final_result_*.json
    if result_dir and os.path.isdir(result_dir):
        for fp in glob(os.path.join(result_dir, "*_final_result_*.json")):
            # sample_id is everything before the first '_final_result_'
            base = os.path.basename(fp)
            sid = base.split("_final_result_")[0]
            if sid:
                done.add(str(sid))

    # 2) Existing shard lines
    if shard_fp and os.path.exists(shard_fp):
        with open(shard_fp, "r", encoding="utf-8") as fin:
            for line in fin:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = obj.get("sample_id")
                if sid is not None:
                    done.add(str(sid))

    # 3) (Optional) per-sample inference traces: {sample_id}_inference_result_*.json
    if log_dir and os.path.isdir(log_dir):
        for fp in glob(os.path.join(log_dir, "*_inference_result_*.json")):
            base = os.path.basename(fp)
            # sample_id is everything before the first '_inference_result_'
            sid = base.split("_inference_result_")[0]
            if sid:
                done.add(str(sid))

    return done

# -----------------------------
# Per-rank worker
# -----------------------------
def _worker_run(
    rank: int,
    n_proc: int,
    vllm_cfg_path: str,
    temperature: float,
    top_p: float,
    top_k: int,
    tool_config_path: Optional[str],
    system_prompt_name: Optional[str],
    max_turns_exp: int,
    max_context_exp: int,
    max_context: int,
    dataset_name: str,
    dataset_split: str,
    item_to_question: str,
    item_to_context: str,
    item_to_answer: str,
    item_to_meta: str,
    correct_answer_key: str,
    model_answer_key: str,
    trajectory_dir: str,
    result_dir: str,
    output_fp: str,
    tokenizer_path: str,
    max_turns_to_fail: int,
    quiet_progress: bool,
    version: str,
    max_items: int = None,
    # NEW:
    resume: bool = False,
    max_output_tokens: int = 4096,
) -> None:
    StateLM, ExecLogger = _import_statelm(version=version)

    vllm_cfg = read_json(vllm_cfg_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # Handle dataset loading for both HuggingFace datasets and local JSON files
    if dataset_name.endswith(('.json', '.jsonl')) and dataset_split == "local":
        data = load_dataset("json", data_files=dataset_name, split="train")
    else:
        data = load_dataset(dataset_name, split=dataset_split)

    # Optional cap: only keep the first `max_items`
    if max_items is not None:
        if max_items <= 0:
            data = data.select([])
        else:
            cap = min(max_items, len(data))
            data = data.select(range(cap))

    item_to_question_fn = _load_callable(item_to_question)
    item_to_context_fn  = _load_callable(item_to_context)
    item_to_answer_fn   = _load_callable(item_to_answer)
    item_to_meta_fn     = _load_callable(item_to_meta) if item_to_meta else None

    # rank-scoped I/O
    rank_suffix = f".rank{rank}"
    out_fp_rank = (
        output_fp.replace(".jsonl", f"{rank_suffix}.jsonl")
        if output_fp.endswith(".jsonl") else
        f"{output_fp}{rank_suffix}.jsonl"
    )

    if trajectory_dir:
        os.makedirs(trajectory_dir, exist_ok=True)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    out_dir = os.path.dirname(out_fp_rank)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # NEW: gather 'done' ids and choose file mode
    done_ids = set()
    if resume:
        done_ids = _collect_done_ids(result_dir=result_dir, log_dir=trajectory_dir, shard_fp=out_fp_rank)
        print(f"[resume][rank {rank}] found {len(done_ids)} completed sample_id(s).")
    fout_mode = "a" if (resume and os.path.exists(out_fp_rank)) else "w"

    skipped = 0
    processed = 0
    with open(out_fp_rank, fout_mode, encoding="utf-8") as fout:
        # Stride sharding: stable & balanced
        for idx in tqdm(range(rank, len(data), n_proc),
                        desc=f"Rank {rank}",
                        position=rank,
                        disable=quiet_progress):
            sample = data[idx]

            # Derive the same sample_id scheme used for saving artifacts
            if "_id" in sample:
                sample_id = str(sample["_id"])
            elif "id" in sample:
                sample_id = str(sample["id"])
            else:
                sample_id = str(idx)

            # Skip if already completed
            if resume and (sample_id in done_ids):
                skipped += 1
                continue
            
            context = item_to_context_fn(sample)
            question = item_to_question_fn(sample)
            correct_ans = item_to_answer_fn(sample)
            meta_dict = item_to_meta_fn(sample) if item_to_meta_fn else {}

            question_type = sample.get("question_type", "Long Context QA")
            
            logger = ExecLogger(log_dir=trajectory_dir, results_dir=result_dir)
            delete_assistant_tool_call_only = version == "niah"
            print(f"[INFO] Setting delete_assistant_tool_call_only: {delete_assistant_tool_call_only}")
            state_lm = StateLM(
                vllm_config=vllm_cfg,
                document_content=context,
                temperature=temperature,
                topp=top_p,
                topk=top_k,
                tool_config_path=tool_config_path,
                system_prompt_name=system_prompt_name,
                tokenizer=tokenizer,
                max_turns_exp=max_turns_exp,
                max_context_exp=max_context_exp,
                max_output_tokens=max_output_tokens,
                delete_assistant_tool_call_only=delete_assistant_tool_call_only
            )
            try:
                last_payload = state_lm.run(question, max_turns_to_fail=max_turns_to_fail)
            except Exception as e:
                print(f"[ERROR][rank {rank}] sample {idx}: {e}")
                last_payload = {"error": str(e)}

            meta_info = {
                "question_type": question_type,
                "sample_id": sample_id,
                "message_count": getattr(state_lm, "ctx_counter", None),
                "notes_count": len(getattr(state_lm.state_manager, "notes", []))
                               if hasattr(state_lm, "state_manager") else None,
                "last_payload": last_payload,
            }
            res_file, final_answer = logger.save_final_result(state_lm, question, correct_ans, meta_info)

            result_info = {
                "question_type": question_type,
                correct_answer_key: correct_ans,
                model_answer_key: final_answer,
                "message_count": getattr(state_lm, "ctx_counter", None),
                "notes_count": len(getattr(state_lm.state_manager, "notes", []))
                               if hasattr(state_lm, "state_manager") else None,
                "last_payload": last_payload,
            }
            inf_file = logger.save_inference_result(question, state_lm, result_info, prefix_tag=sample_id)

            result = {
                "dataset": dataset_name,
                "split": dataset_split,
                "model": "StateLM",
                "sample_id": sample_id,
                "question": question,
                # "context": context,
                "question_type": question_type,
                correct_answer_key: correct_ans,
                model_answer_key: final_answer,
                "meta_info": {
                    "api_call_count": getattr(state_lm, "api_call_counter", 0),
                    "rank": rank,
                    "n_proc": n_proc,
                    "inference_result_path": inf_file,
                    "final_result_path": res_file,
                    "max_turns_to_fail": max_turns_to_fail,
                    "max_output_tokens": max_output_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "system_prompt_name": system_prompt_name,
                    "timestamp": datetime.now().isoformat()
                }
            }

            if meta_dict:
                result.update(meta_dict)
                if 'task_name' in meta_dict: # patch for niah
                    result['others'] = meta_dict
                    result['input'] = question

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            processed += 1

    print(f"[rank {rank}] wrote {out_fp_rank} (processed={processed}, skipped={skipped})")

# -----------------------------
# Public entry
# -----------------------------
def eval_hfds_statelm(
    vllm_cfg: str,
    temperature: float,
    max_turns_exp: int,
    max_context_exp: int,
    max_context: int, 
    tool_config_path: Optional[str],
    system_prompt_name: Optional[str],
    dataset_name: str,
    dataset_split: str,
    item_to_question: str,   # e.g., "my_funcs:item_to_question_fn"
    item_to_context: str,    # e.g., "my_funcs:item_to_context_fn"
    item_to_answer: str,     # e.g., "my_funcs:item_to_answer_fn"
    trajectory_dir: str,
    result_dir: str,
    output_fp: str,
    tokenizer_path: str = "Qwen/Qwen3-8B",
    max_turns_to_fail: int = 80,
    # NEW: internal sharding only
    top_p: float = 1.0,
    top_k: int = None,
    max_output_tokens: int = 4096,
    item_to_meta: str = None,
    output_postprocess: str = None,
    model_answer_key: str = 'final_answer',
    correct_answer_key: str = 'correct_answer',
    n_proc: int = 1,
    max_items: int = None,
    resume: bool = True,
    merge_after: bool = True,
    version: str = "v4",
) -> None:

    if trajectory_dir:
        os.makedirs(trajectory_dir, exist_ok=True)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    if output_fp:
        out_dir = os.path.dirname(output_fp)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    resume_boolean = str(resume).lower() == 'true'
    if n_proc <= 1:
        # single worker behaves like before
        _worker_run(
            rank=0,
            n_proc=1,
            vllm_cfg_path=vllm_cfg,
            temperature=temperature,
            max_turns_exp=max_turns_exp,
            max_context_exp=max_context_exp,
            max_context=max_context,
            tool_config_path=tool_config_path,
            system_prompt_name=system_prompt_name,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            item_to_question=item_to_question,
            item_to_context=item_to_context,
            item_to_answer=item_to_answer,
            item_to_meta=item_to_meta,
            correct_answer_key=correct_answer_key,
            model_answer_key=model_answer_key,
            trajectory_dir=trajectory_dir,
            result_dir=result_dir,
            output_fp=output_fp,
            tokenizer_path=tokenizer_path,
            max_turns_to_fail=max_turns_to_fail,
            quiet_progress=False,
            max_items=max_items,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            resume=resume_boolean,
            version=version
        )
        # For convenience, rename the rank0 shard to the expected output
        if merge_after and output_fp.endswith(".jsonl"):
            shard0 = output_fp.replace(".jsonl", ".rank0.jsonl")
            if os.path.exists(shard0) and shard0 != output_fp:
                os.replace(shard0, output_fp)
                print(f"[merge] single shard → {output_fp}")
        print(f"Final output saved to {output_fp}")
        
        if output_postprocess:
            output_postprocess_fn = _load_callable(output_postprocess)
            output_postprocess_fn(output_fp)
        return

    # Multi-process internal sharding
    mp.set_start_method("spawn", force=True)
    procs = []
    for r in range(n_proc):
        p = mp.Process(
            target=_worker_run,
            kwargs=dict(
                rank=r,
                n_proc=n_proc,
                vllm_cfg_path=vllm_cfg,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
                max_turns_exp=max_turns_exp,
                max_context_exp=max_context_exp,
                max_context=max_context,
                tool_config_path=tool_config_path,
                system_prompt_name=system_prompt_name,
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                item_to_question=item_to_question,
                item_to_context=item_to_context,
                item_to_answer=item_to_answer,
                item_to_meta=item_to_meta,
                correct_answer_key=correct_answer_key,
                model_answer_key=model_answer_key,
                trajectory_dir=trajectory_dir,
                result_dir=result_dir,
                output_fp=output_fp,
                tokenizer_path=tokenizer_path,
                max_turns_to_fail=max_turns_to_fail,
                quiet_progress=(n_proc > 8),
                max_items=max_items,
                resume=resume_boolean,
                version=version
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # Merge shards
    if merge_after and output_fp.endswith(".jsonl"):
        shards = []
        for r in range(n_proc):
            shard = output_fp.replace(".jsonl", f".rank{r}.jsonl")
            if os.path.exists(shard):
                shards.append(shard)
        if shards:
            results = []
            for shard in shards:
                with open(shard, "r", encoding="utf-8") as fin:
                    for line in fin:
                        results.append(json.loads(line))
            # ✅ sort by sample_id
            results.sort(key=lambda x: x.get("sample_id", -1))
            with open(output_fp, "w", encoding="utf-8") as fout:
                for r in results:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[merge] merged {len(shards)} shard(s) → {output_fp} (sorted by sample_id)")
            # Delete rank files after merging
            for shard in shards:
                os.remove(shard)
                print(f"[cleanup] deleted {shard}")
        else:
            print("[merge] no shard files found; nothing to merge.")
    
    if output_postprocess:
        output_postprocess_fn = _load_callable(output_postprocess)
        output_postprocess_fn(output_fp)

# -----------------------------
# Main
# -----------------------------    
if __name__ == "__main__":
    fire.Fire()
