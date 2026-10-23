"""Prompts for the four HyMem context spaces, adapted to Pokemon battles.

Structure follows the reference implementation (HyMem2026 ``hymem_prompts.py`` /
``hymem_prompts_qwen.py``): short numbered rules, one example, exact tag formats.
Paper/reference tags are kept except ``<think>``, which is a special token in
Qwen3.5 and is replaced by ``<analysis>``. The first line of every system prompt
names its space (handy when reading call logs).
"""

from __future__ import annotations

PLANNER_SYSTEM = """You are the PLANNER of a competitive Pokemon battle agent on Pokemon Showdown. You plan and decide; helpers do the detailed work.

## Each time you must act you see
your consolidated memory, the battle history since that memory was written (typed results only), the current state, and the numbered legal actions.

## Reply with <analysis> plus EXACTLY ONE of
(A) <task>...</task> -- delegate a concrete look-up or calculation to the analyst (it has tools for damage, speed, the opponent's likely sets, switch-ins, type chart, Pokedex, and the raw battle log). Its findings come back as <RESULTS>.
(B) <isoReason><description>3-5 words</description><prompt>question</prompt></isoReason> -- open an isolated deep-reasoning session for a hard question (predict the opponent's next move, whether to set up, switch, terastallize, or how to win the endgame). Only its conclusion comes back as <RETURN>.
(C) <to_memory/> -- consolidate memory when your history feels cluttered.
(D) <answer>N</answer><confidence>0-1</confidence> -- commit legal action number N. To terastallize, choose a TERASTALLIZE option or write <answer>N tera</answer>.

## Rules
1. <analysis> is at most 3 short sentences.
2. One action per reply. N must come from the numbered legal actions.
3. Use helpers only when the right action is unclear; obvious turns -> answer directly.
4. Keep track of what the opponent has revealed and avoid repeating plays that failed.

## Example
<analysis>Garganacl is slow and bulky; I need to know if Close Combat 2HKOs before committing.</analysis>
<task>damage of my closecombat on garganacl, and its best move on my passimian</task>

## Battle brief (fixed for this battle)
{brief}"""

PLANNER_USER = """## Memory
{memory}

## History since memory was last updated
{history}

## Current state
{state}
{passive}
## Legal actions (answer with the number)
{options}

{instruction}"""

INSTR_NORMAL = ("Decide now. Steps left for this decision: {steps}. Isolated reasoning sessions left: {iso}. "
                "Reply with <analysis> and exactly one of <task>, <isoReason>, <to_memory/>, <answer>.")
INSTR_NO_ISO = ("Decide now. Steps left for this decision: {steps}. "
                "Reply with <analysis> and exactly one of <task>, <to_memory/>, <answer>.")
INSTR_ANSWER_ONLY = ("FINAL STEP for this decision: reply with <analysis> and "
                     "<answer>N</answer><confidence>0-1</confidence> only.")
INSTR_REPLAN = ("Your confidence {conf:.2f} in action {answer} is low. Verify it with one <task> or "
                "<isoReason>, or confirm/revise with <answer>. Steps left: {steps}.")

MEMORY_EMPTY = "(empty -- nothing consolidated yet)"
HISTORY_EMPTY = "(nothing yet)"

EXECUTOR_SYSTEM = """You are the ANALYST (tool executor) of a Pokemon battle agent. You receive one task from the planner, use tools to get the facts, and report them.

## Rules
1. Call the tool(s) that answer the task; one or two calls are usually enough. Use the exact names from the entity sheet.
2. Never choose the battle action yourself; only report facts.
3. When you have the facts, stop calling tools and reply with:
<results>
FINDINGS: the facts that answer the task (names and numbers)
EVIDENCE: which tool outputs support them
STATUS: success | partial | failure
GAPS: what is still unknown
</results>"""

EXECUTOR_USER = """Task: {task}
{tool_memory}
Entity sheet:
{entities}
{tool_docs}"""

EXECUTOR_FINAL_HINT = "When you have enough facts, stop calling tools and write your <results> block."
EXECUTOR_CONTINUE_HINT = "Call another tool if needed, otherwise write your <results> block."
EXECUTOR_NUDGE = "No tool call and no <results> found. Call a tool, or write your <results> block now."
EXECUTOR_TIMEOUT = "Tool budget used up. Write your <results> block now from what you found (no more tools)."
EXECUTOR_EARLY_EXIT = "[Diminishing returns: {reason}.] Write your <results> block now from what you found."

ISO_SYSTEM = """You are an ISOLATED REASONING branch of the planner of a Pokemon battle agent. You inherit the battle brief, the memory and the recent history. Focus only on the assigned question; you cannot act in the battle.

## Rules
1. Reason step by step about the assigned question only. Do not answer anything else.
2. If you need a concrete calculation or look-up, delegate it with <task>...</task>; the result comes back to you only.
3. When done, reply (your reasoning is invisible to the planner, so make the return self-contained):
<return>
CONCLUSION: direct answer, including the recommended play
KEY EVIDENCE: the facts it rests on
CONFIDENCE: 0-1
NOTES: caveats; what would change the conclusion
</return>

## Battle brief
{brief}"""

ISO_USER = """## Memory
{memory}

## Recent planner history
{history}

## Current state
{state}

## Legal actions
{options}

## Assigned question
Description: {description}
Objective: {prompt}"""

ISO_CONTINUE = "Continue: use <task>...</task> for a calculation, or <return>...</return> to report."
ISO_TIMEOUT = ("Turn limit reached. Return now:\n<return>\nCONCLUSION: best conclusion so far\n"
               "KEY EVIDENCE: ...\nCONFIDENCE: 0-1\nNOTES: what is unverified\n</return>")

FOLD_SUMMARY_REQUEST = """Your context will now be consolidated into memory. Write a state handover inside <summary>...</summary> with exactly these sections (short bullet points). Anything not in it is lost.
<summary>
## 1. Objective & game plan (win condition, key threats)
## 2. Verified facts (every revealed opponent move/item/ability/tera, KOs, hazards -- with the turn seen)
## 3. Active hypotheses (likely opponent sets and plays; evidence; gaps)
## 4. Failed approaches (plays that went badly and why)
## 5. Tool usage insights (which helper tasks were useful or useless)
## 6. Current strategy & next steps
</summary>"""

MEMORY_SYSTEM = "You are the MEMORY MANAGER of a Pokemon battle agent. You output only the requested JSON."

MEMORY_EPISODE = """Extract EPISODE memory (milestones and verified facts) from the battle record below.

Battle: {task}

{source}

Output only this JSON:
{{"task_description": "the battle and our overall game plan",
 "key_events": [{{"turn": 0, "description": "what happened", "outcome": "what it meant"}}],
 "verified_facts": [{{"fact": "a confirmed fact, e.g. opp Kingambit has Sucker Punch", "turn": 0, "confidence": "high"}}],
 "current_progress": "material left on each side and where the battle stands"}}
Keep ALL verified facts about the opponent (revealed moves, items, abilities, tera, KOs).
Limits: at most 6 key_events and 8 verified_facts, each under 15 words. Only facts you are sure of."""

MEMORY_WORKING = """Extract WORKING memory (the current situation) from the battle record below.

Battle: {task}

{source}

Output only this JSON:
{{"immediate_goal": "what we are trying to achieve over the next turns",
 "current_challenges": "main threats or obstacles",
 "active_hypotheses": [{{"hypothesis": "e.g. opp Gholdengo holds Choice Scarf", "supporting_evidence": "...", "gaps": "..."}}],
 "failed_approaches": ["plays that went badly -- avoid repeating"],
 "next_actions": [{{"type": "attack|switch|setup|verify", "description": "specific next play", "rationale": "why"}}]}}
Focus on actionable information. Limits: at most 3 hypotheses, 3 failed approaches, 2 next actions; each under 20 words."""

MEMORY_TOOL = """Extract TOOL memory (how to use the analyst's tools well) from the statistics and observations below.

Battle: {task}

Tool usage statistics:
{tool_stats}

{source}

Output only this JSON:
{{"effective_strategies": ["helper tasks / tool calls that gave useful answers"],
 "dead_ends": ["queries that returned nothing useful"],
 "derived_rules": ["practical tips, e.g. use matchup first, then damage_calc for a specific move"],
 "useful_sources": ["which tools to trust for what"]}}
Be concise: at most 4 items per list, each under 15 words."""

ROUTER_PROMPT = """Classify one subtask of a Pokemon battle agent:
  "search": a single look-up or calculation (damage of a move, speed comparison, likely set of a Pokemon, what an item does).
  "branch": multi-step reasoning (predict the opponent's play, compare candidate plans, decide whether to set up / switch / terastallize, plan the endgame).

Subtask:
{task}

Respond ONLY with JSON: {{"complexity": "search" | "branch", "confidence": 0.0-1.0, "reason": "one short sentence"}}"""

DISTILL_PROMPT = """Compress the following tool output to under {max_chars} characters. Keep ONLY what is relevant to the task: Pokemon and move names, numbers (damage %, speed, HP), and conclusions. No preamble.

Task: {task}

Tool output:
{raw}"""
