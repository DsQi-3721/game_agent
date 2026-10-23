"""System-1 (fast) decision layer: typed, calibrated decisions from a System One model (Jev).

``facts``         battle position -> compact state with code-precomputed semantic facts
``decider``       one typed request per decision: action choice + consistency + complexity
``memory_router`` typed routing of memory items into the contexts that need them
``agents``        ``jev_fast`` (pure System 1) and ``llm_hymem_dual`` (System 1 / HyMem System 2)
"""
