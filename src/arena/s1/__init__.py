"""System-1 learning loop: record -> curate -> distil -> serve -> evaluate -> promote.

Every typed decision the agents ask a System One model (Jev or a local student) goes through
the same TypeSafe-compatible interface, so the loop can swap the model behind it:

  context    who asked, for which decision, for what purpose (contextvars, no plumbing)
  recorder   append-only experience log: every request + answers, the decision it fed, the outcome
  prompts    the ONE rendering of (state, question) into a scorer prompt (server, trainer, dataset)
  scorer     local System One models: option scoring with a causal LM (mlx on Apple silicon, HF/torch
             elsewhere), prefix-cached so all questions of a request share the state's prefill
  server     TypeSafe-compatible HTTP server around a scorer -> TYPESAFE_BASE_URL swaps it in
  dataset    experience -> training examples: teacher soft labels, System-2 corrections, outcome weights
  train_mlx  LoRA distillation on mlx (Mac);  train_hf: the same with torch/peft (GPU servers)
  evaluate   any backend vs held-out teacher labels: agreement, KL, calibration, latency
  registry   model versions, metrics, champion promotion
"""
