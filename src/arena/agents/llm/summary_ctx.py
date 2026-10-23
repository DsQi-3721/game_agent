"""Summary baseline (paper Table 1 "Summary"; ReSum-style recursive summarization).

Like the flat agent, but when the conversation passes ``summary_threshold`` tokens
everything except the last ``keep_last`` decision blocks is folded into a running
summary (recursively: previous summary + the blocks being dropped), which then
lives in the system prompt. A flat context, compressed after the fact -- the
contrast to HyMem's isolation at the source.
"""

from __future__ import annotations

from arena.agents.hymem.messages import clip
from arena.agents.llm.longctx import PersistentContextAgent, estimate_tokens
from arena.llm.spaces import SpaceConfig
from arena.registry.agents import register_agent
from arena.registry.param import Param

SUMMARY_PROMPT = """Summarize this Pokemon battle so far for your future self. Keep: every revealed opponent move / item / ability / tera, KOs, hazards and field effects, what worked and what failed, and the current game plan. Be concise (bullet points).

Previous summary:
{previous}

Conversation to fold into the summary:
{transcript}

Updated summary:"""


@register_agent(
    "llm_summary", is_llm=True, needs_model=True, label="LLM - Summary (recursive summarization)",
    params=[Param("max_steps", int, 4, label="Max steps"),
            Param("summary_threshold", int, 6000, help="summarize when the context exceeds this many tokens",
                  label="Summary threshold"),
            Param("keep_last", int, 3, help="decision blocks kept verbatim", label="Keep last"),
            Param("obs_format", str, "raw", choices=["raw", "digest"], label="Observation"),
            Param("temperature", float, 0.7, label="Temperature")],
)
class SummaryAgent(PersistentContextAgent):
    name = "llm_summary"

    def __init__(self, ctx=None, *, summary_threshold: int = 6000, keep_last: int = 3,
                 summary_max_tokens: int = 512, **kw):
        super().__init__(ctx, **kw)
        self.summary_threshold = int(summary_threshold)
        self.keep_last = max(1, int(keep_last))
        self.summary_sc = SpaceConfig(max_tokens=summary_max_tokens, temperature=0.3, top_p=0.8,
                                      presence_penalty=0.0, thinking=False)

    async def manage_context(self, sess, battle) -> None:
        if estimate_tokens(sess.messages) > self.summary_threshold and len(sess.blocks) > self.keep_last:
            start, end = sess.blocks[0], sess.blocks[-self.keep_last]
            old = sess.messages[start:end]
            transcript = "\n".join(f"[{m['role']}] {clip(str(m.get('content') or ''), 1200)}" for m in old)
            prompt = SUMMARY_PROMPT.format(previous=sess.summary or "(none)", transcript=transcript[-24000:])
            resp = await self._call(sess, [{"role": "user", "content": prompt}], space="summary", sc=self.summary_sc)
            if resp.text.strip():
                sess.summary = clip(resp.text.strip(), 3000)
                del sess.messages[start:end]
                shift = end - start
                sess.blocks = [b - shift for b in sess.blocks[-self.keep_last:]]
                sess.messages[0] = {"role": "system", "content": self._system(sess)}
                sess.summaries += 1
                self._metric("summaries")
        await super().manage_context(sess, battle)   # hard guard
