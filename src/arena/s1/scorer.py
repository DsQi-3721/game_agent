"""Local System-One models: answer typed questions by scoring lettered options with a causal LM.

Each question is rendered by ``prompts.render`` and the next-token distribution over the option
letters becomes the answer's probabilities (softmax with a calibration temperature). No text is
generated, so a request costs one prefill per question; all questions of a request share the
state's prefix, which is prefilled once and reused (prompt-cache copy).

  MlxScorer   Apple silicon (mlx / mlx_lm), optional LoRA adapter (mlx_lm format)
  HFScorer    torch / transformers (+ peft adapter) for CUDA servers; batched, no prefix cache
  ScorerClient  in-process ``system_one`` adapter (evaluation, tests) -- same shape as TypeSafeClient
"""

from __future__ import annotations

import asyncio
import copy
import time
from typing import Optional

from arena.llm.typesafe import parse_result
from arena.s1.prompts import LABELS, render, to_answer


def _common_prefix(seqs: list[list[int]]) -> int:
    if not seqs:
        return 0
    n = min(len(s) for s in seqs)
    first = seqs[0]
    for i in range(n):
        tok = first[i]
        if any(s[i] != tok for s in seqs[1:]):
            return i
    return n


def _chat_ids(tokenizer, messages: list[dict]) -> list[int]:
    try:
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                             enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return list(tokenizer.encode(text, add_special_tokens=False))


def _label_ids(tokenizer) -> list[int]:
    ids = []
    for label in LABELS:
        toks = tokenizer.encode(label, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"label {label!r} is not a single token for this tokenizer: {toks}")
        ids.append(toks[0])
    return ids


class MlxScorer:
    """mlx streams are per thread: the model is loaded and every request is computed on one
    dedicated worker thread, so ``answer`` is safe to call from any thread (event loop, to_thread)."""

    def __init__(self, model_path: str, adapter_path: Optional[str] = None, *, temperature: float = 1.0,
                 model_id: Optional[str] = None):
        from concurrent.futures import ThreadPoolExecutor

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-scorer")
        self._thread = None

        def load():
            import threading

            import mlx_lm
            self._thread = threading.current_thread()
            return mlx_lm.load(model_path, adapter_path=adapter_path)

        self.model, self.tokenizer = self._pool.submit(load).result()
        self.temperature = temperature
        self.model_id = model_id or (f"local:{model_path.rstrip('/').split('/')[-1]}"
                                     + (f"+{adapter_path.rstrip('/').split('/')[-1]}" if adapter_path else ""))
        self.label_ids = _label_ids(self.tokenizer)

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        return _chat_ids(self.tokenizer, messages)

    def answer(self, state, questions: dict) -> tuple[dict, int]:
        import threading
        if threading.current_thread() is self._thread:
            return self._answer(state, questions)
        return self._pool.submit(self._answer, state, questions).result()

    def _answer(self, state, questions: dict) -> tuple[dict, int]:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        items = []
        for qid, q in questions.items():
            msgs, keys = render(state, q)
            items.append((qid, q, keys, self.prompt_ids(msgs)))
        if not items:
            return {}, 0
        lcp = min(_common_prefix([ids for *_, ids in items]), min(len(ids) for *_, ids in items) - 1)
        cache = make_prompt_cache(self.model)
        n_tok = 0
        if lcp > 0:
            self.model(mx.array(items[0][3][:lcp])[None], cache=cache)
            mx.eval([c.state for c in cache])
            n_tok += lcp
        answers = {}
        for i, (qid, q, keys, ids) in enumerate(items):
            c = cache if i == len(items) - 1 else copy.deepcopy(cache)
            logits = self.model(mx.array(ids[lcp:])[None], cache=c)[0, -1]
            lab = logits[mx.array(self.label_ids[:len(keys)])].astype(mx.float32) / self.temperature
            probs = mx.softmax(lab).tolist()
            n_tok += len(ids) - lcp
            answers[qid] = to_answer(q, keys, probs)
        return answers, n_tok


class HFScorer:
    def __init__(self, model_path: str, adapter_path: Optional[str] = None, *, temperature: float = 1.0,
                 model_id: Optional[str] = None, device: Optional[str] = None, batch_size: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else (torch.float16 if self.device == "mps" else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.to(self.device).eval()
        self.temperature = temperature
        self.batch_size = batch_size
        self.model_id = model_id or f"local-hf:{model_path.rstrip('/').split('/')[-1]}"
        self.label_ids = _label_ids(self.tokenizer)

    def answer(self, state, questions: dict) -> tuple[dict, int]:
        torch = self.torch
        items = []
        for qid, q in questions.items():
            msgs, keys = render(state, q)
            items.append((qid, q, keys, _chat_ids(self.tokenizer, msgs)))
        answers, n_tok = {}, 0
        for b in range(0, len(items), self.batch_size):
            chunk = items[b:b + self.batch_size]
            width = max(len(ids) for *_, ids in chunk)
            pad = self.tokenizer.pad_token_id
            input_ids = torch.tensor([[pad] * (width - len(ids)) + ids for *_, ids in chunk], device=self.device)
            mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for *_, ids in chunk], device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=mask).logits[:, -1, :].float()
            for row, (qid, q, keys, ids) in zip(logits, chunk):
                lab = row[torch.tensor(self.label_ids[:len(keys)], device=self.device)] / self.temperature
                answers[qid] = to_answer(q, keys, torch.softmax(lab, dim=-1).tolist())
                n_tok += len(ids)
        return answers, n_tok


class ScorerClient:
    """In-process System-One client around a scorer (serialised: one request at a time)."""

    configured = True

    def __init__(self, scorer):
        self.scorer = scorer
        self._lock = asyncio.Lock()
        self.n_calls = 0
        self.input_tokens = 0
        self.latencies: list[float] = []

    async def system_one(self, state, questions, model=None, headers=None):
        async with self._lock:
            t0 = time.perf_counter()
            answers, n_tok = await asyncio.to_thread(self.scorer.answer, state, questions)
            latency = time.perf_counter() - t0
        self.n_calls += 1
        self.input_tokens += n_tok
        self.latencies.append(latency)
        return parse_result({"model": self.scorer.model_id, "answers": answers, "usage": {"input_tokens": n_tok}},
                            latency)

    async def aclose(self) -> None:
        return None


def load_scorer(backend: str, model_path: str, adapter_path: Optional[str] = None, **kw):
    if backend == "mlx":
        return MlxScorer(model_path, adapter_path, **kw)
    if backend == "hf":
        return HFScorer(model_path, adapter_path, **kw)
    raise ValueError(f"backend must be mlx|hf, got {backend!r}")
