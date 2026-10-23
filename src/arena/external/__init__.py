"""Out-of-process opponent orchestration (PokeChamp / Metamon bridges).

This package only *launches and supervises* external bots; it never imports their
code (which may vendor a conflicting ``poke_env``). The rendezvous is the shared
local Showdown server: our in-process player challenges the external username.
"""
