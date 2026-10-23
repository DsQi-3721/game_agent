"""HyMem battle agent: hierarchical context management via typed context isolation.

Port of HyMem (arXiv 2608.15703; reference implementation HyMem2026/HyMem/agents)
to Pokemon Showdown. Four context spaces -- Planner (the only persistent one),
Executor (tool loop), Isolated reasoning, Fold (memory consolidation) -- that
exchange nothing but typed, schema-bounded messages.
"""
