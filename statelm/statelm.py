from datetime import datetime
import json, re, os, sys, time, uuid
from openai import OpenAI
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional
# --- Elasticsearch ---
# pip install elasticsearch>=9,<10
from elasticsearch import Elasticsearch, helpers
from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError

def read_json(file_path: str) -> Any:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class StateManager:
    def __init__(self): 
        self.notes = {}
        print("[INFO] StateManager Initialized")
    
    def add_note(self, key, content, summary): 
        self.notes[str(key)] = {"summary": str(summary), "full_content": str(content)}
        # print(f"[INFO] snapshot of note library: {self.notes}")
        # print(f"[ACTION]: Note added: Key='{key}', Summary='{summary}'")
        return {"status": "success", "key": key}
    
    def read_note(self, key): 
        # print(f"[ACTION] Reading note: Key='{key}'")
        note = self.notes.get(str(key))
        if note is None:
            return {"error": f"Note '{key}' not found!"}
        return deepcopy(note)   # <-- snapshot, not the live object
    
    def update_note(self, key, mode, new_content=None, new_summary=None):
        k = str(key)
        if k not in self.notes:
            return {"error": f"Note '{key}' not found!"}

        if mode == "delete":
            del self.notes[k]
            return {"status": "success", "key": key, "message": f"Note '{key}' deleted."}
        
        # Only in deletion mode, the new content can be None
        if new_content is None:
            return {"error": "New note content is required."}

        existing_content = self.notes[k]["full_content"]
        if mode == "append":
            full_content = existing_content + "\n" + str(new_content)
            msg = f"Note '{key}' appended."
        elif mode == "overwrite":
            full_content = str(new_content)
            msg = f"Note '{key}' overwritten."
        else:
            return {"error": f"Invalid mode '{mode}'. Use 'append', 'overwrite', or 'delete'."}

        self.notes[k]["full_content"] = full_content
        if new_summary is not None:
            self.notes[k]["summary"] = str(new_summary)
        return {"status": "success", "key": key, "message": msg}
    
    def merge_notes(self, keys, new_key=None, new_summary=None):
        """Merge Multiple Notes into One."""
        notes_to_merge = [] # (key, summary, full_content)
        for key in keys:
            k = str(key)
            if k in self.notes:
                notes_to_merge.append((k, self.notes[k]["summary"], self.notes[k]["full_content"]))
                del self.notes[k]

        if notes_to_merge:
            merged_key = new_key or "_".join([note[0] for note in notes_to_merge])
            existing_summary = new_summary or "  ".join([note[1] for note in notes_to_merge])
            merged_content = "\n".join([note[2] for note in notes_to_merge])

            self.notes[merged_key] = {"summary": str(existing_summary), "full_content": str(merged_content)}
            # print(f"[ACTION]: Notes merged: {merged_keys} -> {new_key}")
            return {"status": "success", "new_key": merged_key, "merged_from": [note[0] for note in notes_to_merge]}

        return {"error": "No notes found to merge."}
    
    def get_notes_summary(self):
        # print("[ACTION]: Getting notes summary")
        if not self.notes: return "No notes recorded."
        return "\n".join([f"- **{key}**: {data['summary']}" for key, data in self.notes.items()])

class ToolLibrary:
    def __init__(self, state_manager, tokenizer, document_content):
        self.state_manager = state_manager
        self.document = document_content
        self.tokenizer = tokenizer
        self.chunk_pointer = [-1, 0]  # (next_forward_chunk_id, next_backward_chunk_id)
        self.last_scanned_chunk_id = -1  # Track last retrieved chunk in scan mode
        self.scan_mode = False
        self.index = []
        self.keywords_searched = set()
        self._es: Optional[Elasticsearch] = None
        self._es_index_name: str = os.getenv('ES_INDEX_NAME', 'lc_agent_document')
        self._es_host: str = os.getenv('ES_HOST', 'https://localhost:9200')
        self._es_user: Optional[str] = os.getenv('ES_USER')
        self._es_pass: Optional[str] = os.getenv('ES_PASS')
        self._es_api_key: Optional[str] = os.getenv('ES_API_KEY')
        self._es_ca_cert: Optional[str] = os.getenv('ES_CA_CERT')
        self.encoded_doc = self.tokenizer(self.document, return_offsets_mapping=True, add_special_tokens=False)
        self._doc_id: Optional[str] = None

        print("[INFO] ToolLibrary Initialized")

    # ---------------------------
    # Elasticsearch helpers (TLS-ready for ES 9.x)
    # ---------------------------
    def _get_es(self) -> Elasticsearch:
        if self._es is None:
            kwargs = {}
            # Auth: prefer API key if provided
            if self._es_api_key:
                kwargs['api_key'] = self._es_api_key
            elif self._es_user and self._es_pass:
                kwargs['basic_auth'] = (self._es_user, self._es_pass)
            # TLS cert
            if self._es_ca_cert:
                kwargs['ca_certs'] = self._es_ca_cert
            self._es = Elasticsearch(self._es_host, **kwargs)
        return self._es

    def _ensure_es_index(self):
        es = self._get_es()
        if es.indices.exists(index=self._es_index_name):
            return
        es.indices.create(
            index=self._es_index_name,
            settings={
                'index': {
                    'analysis': {
                        'analyzer': {
                            'default': {'type': 'standard'}
                        }
                    }
                }
            },
            mappings={
                'properties': {
                    'doc_id':   {'type': 'keyword'},
                    'chunk_id': {'type': 'integer'},
                    'content':  {'type': 'text'},
                    'start_pos': {'type': 'integer'},
                    'end_pos':   {'type': 'integer'}
                }
            }
        )
        print(f"[INFO] Created Elasticsearch index '{self._es_index_name}'.")

    def _bulk_index_chunks(self):
        es = self._get_es()
        actions = ({
            '_op_type': 'index',
            '_index': self._es_index_name,
            '_id': f"{self._doc_id}:{c['chunk_id']}",  # avoid collisions
            'doc_id': self._doc_id, 
            'chunk_id': c['chunk_id'],
            'content': c['content'],
            'start_pos': c['start_pos'],
            'end_pos': c['end_pos'],
        } for c in self.index)
        helpers.bulk(es, actions)
        es.indices.refresh(index=self._es_index_name)
        print(f"[INFO] Indexed {len(self.index)} chunks into Elasticsearch index '{self._es_index_name}'.")

    def clearCurrentDocument(self):
        if not self._doc_id:
            return {"message": "No active document to clear."}
        es = self._get_es()
        es.delete_by_query(
            index=self._es_index_name,
            query={"term": {"doc_id": self._doc_id}},
            refresh=True,
        )
        # reset local state
        self.index = []
        self.keywords_searched = set()
        self._doc_id = None
        self.chunk_pointer = [-1, 0]
        return {"message": "Cleared current document from ES and local state."}

    def analyzeText(self, params):
        return {
            "file_name": "attached_document.txt", 
            "total_tokens": len(self.encoded_doc["input_ids"])
        }

    def buildIndex(self, params):
        chunk_size = params.get("chunk_size", 4000)
        overlap = params.get("overlap", 0)
        
        if chunk_size <= 0: return {"error": "chunk_size must be > 0"}
        if overlap   <  0: return {"error": "overlap must be >= 0"}
        if overlap >= chunk_size: return {"error": "overlap must be < chunk_size"}

        input_ids = self.encoded_doc["input_ids"]
        offsets = self.encoded_doc["offset_mapping"]

        self.index = []
        self._doc_id = uuid.uuid4().hex

        start_token = 0
        chunk_id = 0
        while start_token < len(input_ids):
            end_token = min(start_token + chunk_size, len(input_ids))

            # Get corresponding character positions
            chunk_offsets = offsets[start_token:end_token]
            char_start = chunk_offsets[0][0]
            char_end = chunk_offsets[-1][1]

            # Extract raw text based on char span
            chunk_content = self.document[char_start:char_end]
            chunk_data = {
                "chunk_id": chunk_id,
                "content": chunk_content,
                "start_pos": start_token,
                "end_pos": end_token
            }
            self.index.append(chunk_data)

            chunk_id += 1
            start_token += chunk_size - overlap
        
        self.keywords_searched = set()
        self.chunk_pointer = [-1, 0]  # reset after (re)build

        # Index into Elasticsearch
        try:
            self._ensure_es_index()
            self._bulk_index_chunks()
        except Exception as e:
            return {'error': f'Failed to (re)build Elasticsearch index: {e}'}

        return {
            "index_id": "document_index",
            "total_chunks": len(self.index),
            "first_chunk_id": 0,
            "last_chunk_id": len(self.index) - 1,
        }

    def loadDocument(self, params):
        """Load the full document content."""
        if not self.document:
            return {"error": "Document content is empty."}
        return {
            "document_content": self.document
        }
    
    def readChunk(self, params):
        """
        Retrieve the full text of a document chunk.
        
        Supports two modes:
        1. Explicit mode (enable_scan=false): Retrieve by chunk_id
        2. Scan mode (enable_scan=true): Retrieve next chunk sequentially
        """
        # enable_scan = params.get("enable_scan", False)
        if params.get("enable_scan") is not None:
            self.scan_mode = params.get("enable_scan")
        
        if not self.index:
            return {"error": "Index not built. Please call 'buildIndex' first."}
        
        if self.scan_mode:
            # Scan mode: retrieve the next chunk after the most recently retrieved one
            next_chunk_id = self.last_scanned_chunk_id + 1
            if next_chunk_id >= len(self.index):
                return {"error": f"No more chunks available. Last chunk ({len(self.index)-1}) already retrieved."}
            
            self.last_scanned_chunk_id = next_chunk_id
            return {
                "retrieved_chunk": [self.index[next_chunk_id]], 
                "chunk_id": next_chunk_id,
                "reading_progress": f"{next_chunk_id + 1}/{len(self.index)}"
            }
        else:
            # Explicit mode: retrieve by chunk_id
            chunk_id = params.get("chunk_id")
            if chunk_id is None:
                return {"error": "chunk_id is required when enable_scan is false."}
            
            try:
                chunk_id = int(chunk_id)
            except (ValueError, TypeError):
                return {"error": "chunk_id must be an integer."}
            
            if chunk_id < 0 or chunk_id > (len(self.index)-1):
                return {"error": f"Chunk_id: {chunk_id} is out of range. It must be between 0 and {len(self.index)-1}."}
            
            # Update scan position when explicitly reading a chunk
            self.last_scanned_chunk_id = chunk_id
            return {"retrieved_chunk": [self.index[chunk_id]], "chunk_id": chunk_id}

    def searchEngine(self, params):
        """
        params:
        - keyword: str | list[str]
            Examples:
                "blue mountain, haze"
                ["blue mountain", "haze"]
        - mode: "and" | "or" (default: "or")
            "and": all keywords must appear
            "or": any keyword may appear (default)
        - fragment_size: int (default 180)
        - num_fragments: int (default 3)
        - no_match_size: int (default 120)
        - size: int (default 50)
        - minimum_should_match: str | int (only used for "or", default "1")
        """
        raw_kw = params.get("keyword", "")
        if not self.index:
            return {"error": "Index not built. Please call buildIndex first."}
        if not self._doc_id:
            return {"error": "No active document for this run. Please call buildIndex first."}
        
        # Normalize keywords
        if isinstance(raw_kw, list):
            keywords = [str(k).strip() for k in raw_kw if str(k).strip()]
        else:
            keywords = [k.strip() for k in str(raw_kw).split(",") if k.strip()]

        if not keywords:
            return {"error": "keyword cannot be empty."}

        self.keywords_searched.update(keywords)

        # Using default values for now, not provided by the agent
        mode = (params.get("mode") or "or").lower()
        size = int(params.get("size", 50))
        fragment_size = int(params.get("fragment_size", 180))
        num_frags = int(params.get("num_fragments", 3))
        no_match_size = int(params.get("no_match_size", 120))
        min_should = params.get("minimum_should_match", "1")

        es = self._get_es()

        def _clause(kw: str):
            # Use phrase when there are spaces; slop=2: blue mountain -> mountain blue
            if " " in kw:
                return {"match_phrase": {"content": {"query": kw, "slop": 2}}}
            # Single token: be strict if analyzer splits (operator=and)
            return {"match": {"content": {"query": kw, "operator": "and"}}}
        
        # ---- Build BM25 query ----
        if mode == "and":
            query = {
                "bool": {
                    "must": [_clause(kw) for kw in keywords],
                    "filter": [{"term": {"doc_id": self._doc_id}}],
                }
            }
        elif mode == "or":
            query = {
                "bool": {
                    "should": [_clause(kw) for kw in keywords],
                    "minimum_should_match": min_should,
                    "filter": [{"term": {"doc_id": self._doc_id}}],
                }
            }

        else:
            raise NotImplementedError(f"Search mode '{mode}' not supported.")

        try:
            res = es.search(
                index=self._es_index_name,
                query=query,
                highlight={
                    "pre_tags": ["<em>"],
                    "post_tags": ["</em>"],
                    "fields": {
                        "content": {
                            "type": "unified",
                            "fragment_size": fragment_size,
                            "number_of_fragments": num_frags,
                            "no_match_size": no_match_size
                        }
                    }
                },
                _source=["chunk_id"],
                size=size,
                track_total_hits=False
            )
        except Exception as e:
            return {"error": f"Elasticsearch query failed: {e}"}

        hits = res.get("hits", {}).get("hits", [])
        if not hits:
            return {"retrieved_chunks": [], "message": "No matching content found.", "keywords": keywords}

        items = []
        for h in hits:
            src = h.get("_source", {})
            chunk_id = src.get("chunk_id")
            score = h.get("_score", 0.0)
            highlights = h.get("highlight", {}).get("content", [])
            items.append({
                "chunk_id": chunk_id,
                "relevance_score": round(float(score), 3),
                "highlights": highlights
            })

        items.sort(key=lambda x: x["relevance_score"], reverse=True)
        total = len(items)
        if len(items) > 20:
            items = items[:20]
            return {
                "retrieved_chunks": items,
                "message": f"Showing the most relevant 20/{total} chunks.",
                "keywords": keywords
            }
        return {"retrieved_chunks": items, "keywords": keywords}

    def checkBudget(self, params):
        raise NotImplementedError

    def note(self, params):
        return self.state_manager.add_note(key=params['key'], content=params['content'], summary=params.get('summary'))

    def readNote(self, params):
        return self.state_manager.read_note(key=params['key'])

    def updateNote(self, params):
        return self.state_manager.update_note(
            key=params['key'],
            mode=params.get('mode').lower(),
            new_content=params.get('new_content'),
            new_summary=params.get('new_summary')
        )

    def mergeNotes(self, params):
        return self.state_manager.merge_notes(
            keys=params['keys'],
            new_key=params.get('new_key'),
            new_summary=params.get('new_summary')
        )

    def deleteContext(self, params):
        raise NotImplementedError

    def getContextStats(self, params):
        """Get context statistics."""
        return {
            "total_notes": len(self.state_manager.notes),
            "notes_keys": list(self.state_manager.notes.keys()),
            "index_chunks": len(self.index),
            "document_size": len(self.encoded_doc["input_ids"]),
            "searched_keywords": list(self.keywords_searched),
        }

    def finish(self, params):
        return {"final_answer": params.get("answer", "No final answer provided.")}


class ExecLogger:
    """Execution Logger for saving query logs and inference results"""
    def __init__(self, log_dir="logs", results_dir="results"):
        self.log_dir = log_dir
        self.results_dir = results_dir
        self.ensure_output_dir()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.api_calls = []
        
    def ensure_output_dir(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def log_api_call(self, api_input, api_output, call_index):
        api_call = {
            "timestamp": datetime.now().isoformat(),
            "call_index": call_index,
            "session_id": self.session_id,
            "api_input": api_input,
            "api_output": api_output
        }
        self.api_calls.append(api_call)
        print(f"[INFO] API call {call_index} has been recorded")
    
    def save_query_log(self, query, document_info, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "query": query,
            "document_info": document_info,
            "api_calls_count": len(self.api_calls)
        }
        
        log_file = os.path.join(self.log_dir, f"query_log_{self.session_id}.json")
        
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Query log saved to: {log_file}")

    def save_inference_result(self, query, orchestrator, result_info=None, prefix_tag=None):
        timestamp = datetime.now().isoformat()

        # Sanitize everything that might contain SDK objects
        sanitized_history = orchestrator._sanitize_for_json(orchestrator.full_history)
        sanitized_result_info = orchestrator._sanitize_for_json(result_info or {})

        inference_trace = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "system_prompt": orchestrator.system_prompt,
            "query": query,
            "result_info": sanitized_result_info,
            "full_history": sanitized_history,
        }

        # (rest unchanged)
        result_file = os.path.join(self.log_dir, f"inference_result_{self.session_id}.json")
        if prefix_tag:
            result_file = os.path.join(self.log_dir, f"{prefix_tag}_inference_result_{self.session_id}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(inference_trace, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Inference results saved to: {result_file}")
        return result_file

    
    def save_api_calls_log(self):
        """Save API calls log separately"""
        if not self.api_calls:
            return
        
        api_log_file = os.path.join(self.log_dir, f"api_calls_{self.session_id}.json")
        with open(api_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.api_calls, f, ensure_ascii=False, indent=2)

        print(f"[INFO] API calls log saved to: {api_log_file}")

    def save_final_result(self, orchestrator, question, expected_answer, meta_info=None, filename=None):
        """Save the final results and metadata to the results directory"""
        full_history = orchestrator.full_history
        final_answer = None
        for msg in reversed(full_history):
            if msg.get("role") == "tool":
                final_answer = msg.get("content", {}).get("final_answer")
                if final_answer:
                    break
        if not final_answer:
            final_answer = "No final answer found."
        result_info = {
            "session_id": self.session_id,
            "question": question,
            "final_answer": final_answer,
            "expected_answer": expected_answer,
            "meta_info": meta_info or {}
        }
        if filename:
            result_file = os.path.join(self.results_dir, filename)
        else:
            sample_id = (meta_info or {}).get("sample_id", "id_unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(self.results_dir, f"{sample_id}_final_result_{timestamp}.json")

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_info, f, ensure_ascii=False, indent=4)
        print(f"[INFO] Final results saved to: {result_file}")
        return result_file, final_answer

class StateLM:
    def __init__(self, 
                 vllm_config: Dict[str, Any], 
                 document_content: str, 
                 temperature: float, 
                 tokenizer: Any, 
                 logger: Optional[ExecLogger] = None, 
                 max_context_exp: int = 30720, 
                 max_turns_exp: int = 50, 
                 max_output_tokens: int = 4096, 
                 system_prompt_name: Optional[str] = None, 
                 tool_config_path: Optional[str] = None,
                 topp: float = 1.0,
                 topk: int = None,
                 delete_assistant_tool_call_only: bool = False,
                ):
        print("[INFO] Setting up OpenAI Client...")
        
        # Model & client config
        self.model_name = (
            vllm_config.get("MODEL_ID", "StateLM-8B")
        )

        openai_base = vllm_config.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        openai_key  = vllm_config.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY is required. Set env or pass via vllm_config['OPENAI_API_KEY'].")

        if openai_base:
            self.vllm_client = OpenAI(api_key=openai_key, base_url=openai_base)
        else:
            self.vllm_client = OpenAI(api_key=openai_key)

        self.system_prompt = self._get_system_prompt_text(system_prompt_name)
        # print(f"[INFO] System prompt set as '{self.system_prompt[:-100]}...'")
        self.tools = self._get_tool_config(tool_config_path)
        print(f"[INFO] OpenAI Client ready for model '{self.model_name}' with {len(self.tools)} tools configured.")
        self.tool_names = [item['function']['name'] for item in self.tools]

        # Runtime state (unchanged)
        self.state_manager = StateManager()
        self.tool_library = ToolLibrary(self.state_manager, tokenizer, document_content)
        self.tokenizer = tokenizer
        self.full_history: List[Dict[str, Any]] = []
        self.ctx_counter = 0  # unified counter for assistant/tool messages
        self.deleted_msg_ids = set()   # contains msg_id of both assistant & tool messages

        self.logger = logger
        self.api_call_counter = 0
        self.temperature = temperature
        self.max_context_exp = max_context_exp
        self.max_turns = max_turns_exp
        self.max_output_tokens = max_output_tokens
        self.topp = topp
        self.topk = topk
        self.delete_assistant_tool_call_only = delete_assistant_tool_call_only

    def _get_system_prompt_text(self, system_prompt_name=None):
        if system_prompt_name is None:
            system_prompt_name = "STATELM_SYSTEM_PROMPT"
        from LC_Agent.tools_and_prompt import prompts
        system_prompt = getattr(prompts, system_prompt_name)
        print(f"[INFO] Using system prompt: {system_prompt_name}")
        return system_prompt

    def _get_tool_config(self, tool_config_path=None):
        if tool_config_path:
            print(f"[INFO] Using custom tool config: {tool_config_path}")
            return read_json(tool_config_path)
        print(f"[INFO] Using default tool config: statelm_tools.json")
        return read_json("../tools_and_prompt/statelm_tools.json")

    def _resolve_msg_entry(self, msg_id: int):
        for i, m in enumerate(self.full_history):
            if m.get("msg_id") == msg_id:
                return i, m
        return None, None

    def _parse_llm_output(self, resp):
        choice = resp.choices[0]
        finish_reason = choice.finish_reason
        msg = choice.message
        text = (msg.content or "").strip()           # assistant plain text
        tool_calls = msg.tool_calls or []            # list[ChatCompletionMessageToolCall]
        # print(f"[DEBUG] text: {text}")
        # print(f"[DEBUG] finish_reason: {finish_reason}")

        if finish_reason == "tool_calls" and tool_calls:
            call = tool_calls[0]                     # we allow one call per step
            #print(f"[DEBUG] tool call: {call}")
            return (
                text,                                # assistant response (no <thought>)
                call.function.name,                  # tool name
                json.loads(call.function.arguments), # dict of arguments
                call.id,                             # tool_call_id
                finish_reason,
            )

        return text, None, None, None, finish_reason

    def _build_api_payload(self):
        """
        Build a list[dict] `messages` that conforms to the OpenAI / vLLM format.

        * role == "system"  ➜ first element, contains the long system prompt
        * user / assistant  ➜ strings; assistant may also include `tool_calls`
        * tool results      ➜ role == "tool" (must carry the matching tool_call_id)
        """
        STUB_MESSAGE = "Content has been deleted to save space."
        
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        notes_summary = (
            f"\n\n<external_memory>\n## Available Notes\n"
            f"{self.state_manager.get_notes_summary()}\n</external_memory>"
        )

        for idx, msg in enumerate(self.full_history):
            role = msg.get("role")

            if role == "user":
                text = msg["content"] + (notes_summary if idx == 0 else "")
                messages.append({"role": "user", "content": text})

            elif role == "assistant":
                msg_id = msg["msg_id"]
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    normalized_tool_calls = []
                    for tc in tool_calls:
                        if hasattr(tc, "model_dump"):
                            normalized_tool_calls.append(tc.model_dump())
                        elif isinstance(tc, dict):
                            normalized_tool_calls.append(tc)
                        else:
                            normalized_tool_calls.append(json.loads(json.dumps(tc, default=str)))

                if msg_id in self.deleted_msg_ids:
                    tool_calls = msg.get("tool_calls") or []
                    if tool_calls:
                        # Normalize all tool calls (objects -> dicts) and build valid stubs
                        stubs = []
                        for tc in tool_calls:
                            if hasattr(tc, "model_dump"):
                                tcd = tc.model_dump()
                            elif isinstance(tc, dict):
                                tcd = tc
                            else:
                                tcd = json.loads(json.dumps(tc, default=str))

                            fn = tcd.get("function") or {}
                            name = fn.get("name") or ""

                            stubs.append({
                                "id": tcd.get("id"),
                                "type": "function",
                                "function": {
                                    "name": name,
                                    # arguments MUST be a JSON string
                                    "arguments": json.dumps(
                                        {"message": STUB_MESSAGE},
                                        ensure_ascii=False,
                                    ),
                                },
                            })
                        # messages.append({
                        #     "role": "assistant",
                        #     "content": "Content has been deleted to save space.",
                        #     "tool_calls": stubs,
                        # })
                        if self.delete_assistant_tool_call_only:
                            raw_text = " ".join(
                                blk.get("text", "")
                                for blk in msg.get("content", [])
                                if blk.get("type") == "text"
                            )
                        else:
                            raw_text = STUB_MESSAGE

                        messages.append({
                            "role": "assistant",
                            "content": raw_text.strip() if raw_text.strip() else "",
                            "tool_calls": stubs
                        })
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": STUB_MESSAGE,
                        })

                    continue

                # Not deleted messages
                raw_text = " ".join(
                    blk.get("text", "")
                    for blk in msg.get("content", [])
                    if blk.get("type") == "text"
                )
                # Remove ONLY the <think> and </think> tags (keep inner content)
                # cleaned_text = re.sub(r"</?think>", "", raw_text, flags=re.IGNORECASE).strip()
                cleaned_text = raw_text.strip()
                assistant_msg = {
                    "role": "assistant",
                    # Match OpenAI's acceptance of None for empty text (like your sample)
                    "content": (cleaned_text if cleaned_text else None),
                }
                # If this assistant turn contained tool calls, forward them verbatim
                if tool_calls:
                    assistant_msg["tool_calls"] = normalized_tool_calls
                messages.append(assistant_msg)

            elif role == "tool":
                msg_id = msg["msg_id"]
                msg_id_ia = msg["msg_id(invoking_assistant)"]
                tool_use_id = msg["tool_use_id"]
                tool_result_content_cp = deepcopy(msg["content"])  # 深拷贝以避免修改原始内容
                tool_result_content_cp["msg_id"] = msg_id    # we need to ensure msg_id is included in the result
                tool_result_content_cp["msg_id(invoking_assistant)"] = msg_id_ia
                if msg_id in self.deleted_msg_ids:
                    # if "retrieved_chunks" in tool_result_content_cp:
                    #     # 如果是检索结果，删除内容以节省空间
                    #     tool_result_content_cp["retrieved_chunks"] = []
                    #     tool_result_content_cp["status"] = "success"
                    #     tool_result_content_cp["message"] = "Content has been deleted to save space."
                    #     tool_result_content_cp["original_tool"] = msg.get("tool_name", "unknown")
                    tool_name = msg.get("tool_name", "unknown")
                    if tool_name not in ["nextChunk", "readChunk", "note", "updateNote"]:
                        print(f"[INFO] Attempting to delete {msg.get('tool_name', 'unknown')}")
                    tool_result_content_cp = {
                        "msg_id": msg_id,
                        "msg_id(invoking_assistant)": msg_id_ia,
                        "status": "success",
                        "message": STUB_MESSAGE,
                        "original_tool": msg.get("tool_name", "unknown")
                    }
                    if msg.get("tool_name") in ["nextChunk", "readChunk"]:
                        if "reading_progress" in msg["content"]:
                            tool_result_content_cp["reading_progress"] = msg["content"]["reading_progress"]
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(tool_result_content_cp, ensure_ascii=False),
                        "tool_call_id": tool_use_id,
                    }
                )
        return messages

    def _call_llm_api(self, messages):
        body_kwargs = {
            "model": self.model_name,      # must match vLLM --served-model-name
            "messages": messages,
            "tools": self.tools,
            "temperature": self.temperature,
            "top_p": self.topp,
            "max_tokens": self.max_output_tokens,  # Change to max_completion_tokens in for OpenAI
            # "extra_body": {"repetition_penalty": 1.0, "top_k": self.topk},
        }
        if self.topk:
            body_kwargs["extra_body"] = {"top_k": self.topk}  # vllm_only

        # print(f"[DEBUG] LLM API Payload (without tools and messages):")
        # debug_payload = {k: v for k, v in body_kwargs.items() if k not in ["tools", "messages"]}
        # print(json.dumps(debug_payload, indent=2))
        
        # tries, max_tries = 0, 1
        tries, max_tries = 0, 0
        while True:
            try:
                resp = self.openai_client.chat.completions.create(**body_kwargs)
                self.api_call_counter += 1
                if getattr(self, "logger", None):
                    self.logger.log_api_call(body_kwargs, resp.model_dump(), self.api_call_counter)
                return resp
            except (APIError, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as e:
                if tries >= max_tries:
                    raise
                wait = 5 * (2**tries)
                print(f"[API] {e} - retrying in {wait}s")
                time.sleep(wait)
                tries += 1



    def _execute_tool(self, action, params):
        if action == "checkBudget" or action == "getContextStats":
            messages = self._build_api_payload()
            # messages = convert_to_tokenizer_format(api_payload)
            tokenized_messages = self.tokenizer.apply_chat_template(messages, tools=self.tools, add_generation_prompt=False, tokenize=True)
            conv_rounds = len(self.full_history) // 2
            message_len = len(tokenized_messages) # tool specs involved
            budget_info =  {
                "conv_rounds": conv_rounds,
                "available_tokens": max(self.max_context_exp - message_len - self.max_output_tokens, 0),
                "available_rounds": max(self.max_turns - conv_rounds, 0),
            }
            if action == "checkBudget":
                return budget_info
            else:
                context_stats = self.tool_library.getContextStats(params or {})
                context_stats.update(budget_info)
                return context_stats

        if action == "deleteContext":
            msg_id = params.get("msg_id")
            if msg_id is None:
                return {"error": "msg_id is required"}
            idx, entry = self._resolve_msg_entry(int(msg_id))
            if entry is None:
                return {"error": f"msg_id {msg_id} not found"}
            role = entry.get("role")
            if role == "user":
                return {"error": "Deleting user messages is not supported"}
            elif role in ("assistant", "tool"):
                self.deleted_msg_ids.add(int(msg_id))
                return {"status": "success", "deleted_msg_id": int(msg_id), "deleted_role": role}
            return {"error": f"Unsupported role '{role}' for deletion"}

        if hasattr(self.tool_library, action):
            return getattr(self.tool_library, action)(params or {})
        return {"error": f"Tool '{action}' not found."}

    # ------- Wrap base run with Ctrl-C handling only -------
    def run(self, user_query, max_turns_to_fail=80):
        """
        Self-contained OpenAI/vLLM loop so base Orchestrator never touches
        the raw SDK response (avoids resp.get() errors).
        """
        # Ensure a first user turn with msg_id
        self.full_history.append({"role": "user", "content": user_query})
        self.ctx_counter = 0
        turn = 0
        try:
            while turn <= max_turns_to_fail: 
                print(f"\n--- Round {turn} (Max {max_turns_to_fail} rounds, expected within {self.max_turns} rounds) ---")
                api_payload = self._build_api_payload()
                try:
                    resp = self._call_llm_api(api_payload)
                except Exception as e:
                    err = f"LLM API failed after retries: {type(e).__name__}: {e}"
                    print("[ERROR]", err)
                    self.full_history.append({
                        "role": "tool",
                        "content": {"status": "error", "message": err},
                        "msg_id": self.ctx_counter + 1,
                        "msg_id(invoking_assistant)": self.ctx_counter,
                        "tool_use_id": "api_failure",
                        "tool_name": "finish"
                    })
                    self.tool_library.clearCurrentDocument()
                    return api_payload
                
                self.ctx_counter += 1
                thought, action, params, tool_use_id, stop_reason = self._parse_llm_output(resp)
                msg_id = self.ctx_counter

                self.full_history.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": resp.choices[0].message.content or ""}],
                    "tool_calls": resp.choices[0].message.tool_calls,  # <-- keep tool_calls for replay
                    "msg_id": msg_id
                })
                print("[RUN] Assistant:", thought)

                if stop_reason == 'tool_calls':
                    print(f"[RUN] Assistant action: Call tool `{action}`, parameters: {params}")
                    # self.tool_call_counter += 1
                    
                    if action not in self.tool_names:
                        result = {"error": f"Tool '{action}' not found."}
                    else:
                        result = self._execute_tool(action, params)

                    self.ctx_counter += 1
                    msg_id_tool = self.ctx_counter
                    self.full_history.append({
                        "role": "tool", 
                        "content": deepcopy(result),
                        "msg_id": msg_id_tool,
                        "msg_id(invoking_assistant)": msg_id,
                        "tool_use_id": tool_use_id,
                        "tool_name": action
                    })
                    
                    # 限制输出长度
                    result_preview = json.dumps(result, ensure_ascii=False)
                    if len(result_preview) > 200:
                        result_preview = result_preview[:200] + "..."
                    print(f"[RUN] Tool result (ID: {msg_id_tool}): {result_preview}")

                    if action == "finish":
                        print(f"\n--- Final Answer --- \n{result.get('final_answer', 'No final answer provided.')}")
                        break
                
                else:
                    print(f"[INFO] Process terminated due to stop_reason '{stop_reason}'.")
                    break

                turn += 1
            
            if turn > self.max_turns:
                print(f"[INFO] Reached max rounds {self.max_turns}, stopping execution.")
            self.tool_library.clearCurrentDocument()
            return self._build_api_payload() # full api payload for logging

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user (Ctrl-C). Exiting gracefully...")
            # self.save_trajectory(out_dir="logs", filename="trajectory_interrupted.json")
            self.tool_library.clearCurrentDocument()
            return self._build_api_payload()

    # ------- Minimal extract_final_answer passthrough if base has one -------
    # (keeps behavior but skips deleted assistant turns)
    def _extract_final_answer(self):
        # Prefer a finish tool result if present
        for msg in reversed(self.full_history):
            if msg.get("role") == "tool" and msg.get("tool_name") == "finish":
                content = msg.get("content", {})
                if isinstance(content, dict) and "final_answer" in content:
                    return content.get("final_answer")
        # Else the last non-deleted assistant text
        for msg in reversed(self.full_history):
            if msg.get("role") == "assistant" and msg.get("msg_id") not in self.deleted_msg_ids:
                text = ""
                for blk in (msg.get("content") or []):
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        text += blk.get("text") or ""
                if (text := text.strip()):
                    return text
        return ""

    # ---------- Built-in: save trajectory & final answer ----------
    def _sanitize_for_json(self, obj):
        """Recursively convert SDK / complex objects to plain JSON-safe types."""
        # 1) Try model_dump()/dict()
        if hasattr(obj, "model_dump"):
            obj = obj.model_dump()
        elif hasattr(obj, "dict"):
            obj = obj.dict()

        # 2) Recurse containers
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._sanitize_for_json(v) for v in obj]

        # 3) Handle bytes-like
        if isinstance(obj, (bytes, bytearray)):
            try:
                return obj.decode("utf-8", errors="replace")
            except Exception:
                return str(obj)

        # 4) Last-resort: ensure jsonable via round-trip
        try:
            json.dumps(obj, ensure_ascii=False)
            return obj
        except TypeError:
            return str(obj)

    # ------- Save trajectory override (adds deleted ids) -------
    def save_trajectory(self, out_dir="logs", filename=None, correct_answer=None, meta_info=None):
        snapshot = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "model": getattr(self, "model_name", "Qwen3-8B-Agentic"),
            "session_notes_summary": self.state_manager.get_notes_summary(),
            "api_call_count": self.api_call_counter,
            "final_answer": self._extract_final_answer(),
            "correct_answer": correct_answer,
            "full_history": self._sanitize_for_json(self.full_history),
            "deleted_msg_ids": sorted(self.deleted_msg_ids),
            "meta_info": meta_info,
        }

        os.makedirs(out_dir or ".", exist_ok=True)
        if not filename:
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            filename = f"trajectory_{ts}.json"
        path = os.path.join(out_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Trajectory saved to: {path}")
        return path