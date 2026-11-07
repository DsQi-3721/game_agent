"""Local System-One models: answer typed questions by scoring lettered options with a causal LM.

Each question is rendered by ``prompts.render`` and the next-token distribution over the option
letters becomes the answer's probabilities (softmax with a calibration temperature; the action
questions can have their own, ``action_temperature``, as one temperature rarely fits every question
type: see ``question_temperature``). No text is
generated, so a request costs one prefill per question; all questions of a request share the
state's prefix, which is prefilled once and reused (prompt-cache copy).

  MlxScorer   Apple silicon (mlx / mlx_lm), optional LoRA adapter (mlx_lm format)
  HFScorer    torch / transformers (+ peft adapter) for CUDA servers: all questions of a request
              in one right-padded batch through ``label_logits`` (the trainer's exact computation)
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


def _chat_ids_batch(tokenizer, conversations: list[list[dict]]) -> list[list[int]]:
    """Batch tokenization with the same template and special-token rules as single requests."""
    try:
        texts = tokenizer.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False,
                                              enable_thinking=False)
    except TypeError:
        texts = tokenizer.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    return tokenizer(texts, add_special_tokens=False)["input_ids"]


def label_logits(model, seqs: list[list[int]], label_ids: list[int], pad_id: int, *, return_hidden=False,
                 pad_to_length=None, pad_to_batch=None):
    """torch: logits over the option letters at each sequence's last token -> (len(seqs), len(label_ids)) fp32.

    The batch is right-padded and the model is causal, so no position before the padding ever sees
    it: no attention mask is needed and every row equals scoring that sequence alone (checked in
    fp32 on Qwen3.5, linear-attention layers included: |dp| < 5e-4). Only the label rows of the
    output head are computed -- the full-vocabulary logits of every position (B x T x 248k for
    Qwen3.5, ~4 GiB per batch of 8) are never materialised. Shared by ``HFScorer`` and
    ``train_hf`` so serving and training cannot drift apart."""
    import torch

    base = model.get_base_model() if hasattr(model, "get_base_model") else model     # peft wrapper
    head = base.get_output_embeddings()
    dev = head.weight.device
    n = len(seqs)
    if pad_to_batch and n < pad_to_batch:
        seqs = seqs + [seqs[-1]] * (pad_to_batch-n)
    width = max(max(len(s) for s in seqs), pad_to_length or 0)
    x = torch.tensor([s + [pad_id] * (width - len(s)) for s in seqs], device=dev)
    last = torch.tensor([len(s) - 1 for s in seqs], device=dev)
    hidden = base.get_decoder()(input_ids=x).last_hidden_state
    h = hidden[torch.arange(len(seqs), device=dev), last].float()
    rows = torch.as_tensor(label_ids, device=dev)
    out = h @ head.weight[rows].float().T
    if getattr(head, "bias", None) is not None:
        out = out + head.bias[rows].float()
    return (out[:n], h[:n]) if return_hidden else out[:n]


# question ids of an action request (``agents.system1.decider``); "<qid>_rev" is the reversed-order copy
ACTION_QIDS = ("best", "best_move", "best_switch", "stay", "hard")


def question_temperature(qid: str, temperature: float, action_temperature: Optional[float] = None) -> float:
    """The calibration temperature of one question: ``action_temperature`` (when set) for the action
    questions, ``temperature`` for everything else (memory, step control, task route, tool plan, general).
    s1-v7-2b: 1.55 fits the action questions, but the others are best calibrated near 1
    (reports/fusion23_qwen27b.md §7.2)."""
    if action_temperature is not None and qid.removesuffix("_rev") in ACTION_QIDS:
        return action_temperature
    return temperature


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
                 action_temperature: Optional[float] = None, model_id: Optional[str] = None):
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
        self.action_temperature = action_temperature
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
            lab = (logits[mx.array(self.label_ids[:len(keys)])].astype(mx.float32)
                   / question_temperature(qid, self.temperature, self.action_temperature))
            probs = mx.softmax(lab).tolist()
            n_tok += len(ids) - lcp
            answers[qid] = to_answer(q, keys, probs)
        return answers, n_tok


class HFScorer:
    """torch / transformers (+ peft adapter). Questions are sorted by length and scored in batches
    of ``batch_size`` (an action request -- 5-7 questions -- is one forward pass). The adapter is
    applied unmerged by default, exactly as it was trained; ``merge`` folds it into the base
    weights (fewer kernels, but bf16 rounding of the merged weights is slightly less faithful)."""

    def __init__(self, model_path: str, adapter_path: Optional[str] = None, *, temperature: float = 1.0,
                 action_temperature: Optional[float] = None, model_id: Optional[str] = None,
                 device: Optional[str] = None, batch_size: int = 16, dtype: Optional[str] = None,
                 merge: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        if dtype:
            tdtype = getattr(torch, dtype)
        else:
            tdtype = torch.bfloat16 if self.device == "cuda" else (torch.float16 if self.device == "mps"
                                                                   else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pad_id = (self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                       else self.tokenizer.eos_token_id)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=tdtype)
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            if merge:
                model = model.merge_and_unload()
        self.model = model.to(self.device).eval()
        self.temperature = temperature
        self.action_temperature = action_temperature
        self.batch_size = batch_size
        self.model_id = model_id or (f"local-hf:{model_path.rstrip('/').split('/')[-1]}"
                                     + (f"+{adapter_path.rstrip('/').split('/')[-1]}" if adapter_path else ""))
        self.label_ids = _label_ids(self.tokenizer)

    def answer(self, state, questions: dict) -> tuple[dict, int]:
        return self.answer_many([(state, questions)])[0]

    def answer_many(self, requests: list[tuple]) -> list[tuple[dict, int]]:
        """Several requests at once: all their questions are length-sorted and scored together in batches
        of ``batch_size`` (rows are independent, so the answers equal scoring each request alone)."""
        torch = self.torch
        items = []
        for r, (state, questions) in enumerate(requests):
            for qid, q in questions.items():
                msgs, keys = render(state, q)
                items.append((r, qid, q, keys, _chat_ids(self.tokenizer, msgs)))
        items.sort(key=lambda it: len(it[4]))
        answers: list[dict] = [{} for _ in requests]
        n_tok = [0] * len(requests)
        with torch.inference_mode():
            for b in range(0, len(items), self.batch_size):
                chunk = items[b:b + self.batch_size]
                logits = label_logits(self.model, [ids for *_, ids in chunk], self.label_ids, self.pad_id)
                for row, (r, qid, q, keys, ids) in zip(logits, chunk):
                    t = question_temperature(qid, self.temperature, self.action_temperature)
                    probs = torch.softmax(row[:len(keys)] / t, dim=-1).tolist()
                    answers[r][qid] = to_answer(q, keys, probs)
                    n_tok[r] += len(ids)
        return [({qid: answers[r][qid] for qid in questions}, n_tok[r]) for r, (_s, questions) in enumerate(requests)]


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
