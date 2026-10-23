"""Offline game knowledge: Dex text, opponent set priors, stat estimates.

Built once by ``scripts/build_knowledge.py`` into ``data/knowledge/*.json``; loaded
lazily (constructing a tool never does IO) and cached per process.
"""
