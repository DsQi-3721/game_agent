"""Player observations with explicit knowledge semantics, raw stream capture and simulator replay.

- ``schema``  -- PlayerObservation (model input) and DecisionTrace (what happened to a decision), with
  knowledge classes and provenance (schema ``obs-1``).
- ``capture`` -- per-player raw stream recorder used by ``AgentPlayer`` (everything received / sent).
- ``replay``  -- regenerate both players' exact streams from a Showdown server log (seed, teams,
  inputLog) and compare them with a live capture.
"""
