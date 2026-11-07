"""Tournament / cross-play evaluation.

Builds agents from config, runs the pairing schedule (round-robin or anchored),
enforces guardrails between battles, and produces a report: win-rate matrix,
per-agent win rate + Wilson CI, Elo, behavioral + cost metrics. Matchups run
sequentially (one challenge per username pair); per-agent trajectories stream to
JSONL as they play.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from arena.agents.base import AgentContext
from arena.agents.registry import build_spec, is_llm_type
from arena.augment.base import AugmentationRegistry
from arena.config.schema import ExperimentConfig
from arena.eval.guardrails import BudgetExceeded, Guardrails
from arena.eval.metrics import elo_from_results, wilson_interval
from arena.eval.runner import MatchRunner
from arena.events.emitter import EventEmitter
from arena.events.schema import EventType
from arena.logging.trajectory import TrajectoryLogger
from arena.state.renderer import DefaultRenderer
from arena.teams.manager import AssignmentMode, TeamManager
from arena.teams.pool import DirTeamPool


def merge_extra_metrics(into: dict, new: Optional[dict]) -> dict:
    """Fold one agent instance's ``extra_metrics()`` into the running totals.

    Numeric values add up across pairings, except keys ending in ``_max`` which keep
    the maximum. Averages are derived later from the ``*_sum`` / count keys.
    """
    for k, v in (new or {}).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if k.endswith("_max"):
            into[k] = max(into.get(k, v), v)
        else:
            into[k] = into.get(k, 0) + v
    return into


def _scalar_extra(extra) -> dict:
    """Numeric/bool subset of a per-decision ``extra`` dict (kept small for events)."""
    if not isinstance(extra, dict):
        return {}
    return {k: v for k, v in extra.items() if isinstance(v, (int, float, bool))}


def team_blocks(n_battles: int, per_team: int, offset: int, n_teams: int) -> list[tuple[int, int]]:
    """Mirror-team blocks of one pairing -> [(team index, battles)]. Battle g of the global sequence (this run's
    battles are g = offset .. offset + n_battles - 1) plays team (g // per_team) % n_teams; consecutive battles
    with the same team form one block (one match). ``per_team`` 0: a single block, the runner draws the team."""
    if per_team <= 0 or n_teams <= 0:
        return [(-1, n_battles)] if n_battles > 0 else []
    out: list[tuple[int, int]] = []
    for g in range(offset, offset + n_battles):
        t = (g // per_team) % n_teams
        if out and out[-1][0] == t:
            out[-1] = (t, out[-1][1] + 1)
        else:
            out.append((t, 1))
    return out


def pool_blocks(n_battles: int, per_team: int, offset: int, n_a: int, n_b: int) -> list[tuple[tuple[int, int], int]]:
    """Two-pool blocks -> [((our team, their team), battles)]: battle g plays block b = g // per_team, our team
    b % n_a against their team b % n_b."""
    per_team = max(per_team, 1)
    out: list = []
    for g in range(offset, offset + n_battles):
        b = g // per_team
        key = (b % n_a, b % n_b, b)
        if out and out[-1][0] == key:
            out[-1] = (key, out[-1][1] + 1)
        else:
            out.append((key, 1))
    return [((a, t), nb) for (a, t, _b), nb in out]


class Evaluator:
    def __init__(self, config: ExperimentConfig, *, llm_client=None, run_dir: Path | str = "runs",
                 watch: bool = False, turn_delay: float = 0.0,
                 emitter: Optional[EventEmitter] = None, metrics_sink=None):
        self.cfg = config
        self.client = llm_client
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.watch = watch
        self.turn_delay = turn_delay
        self.emitter = emitter or EventEmitter.disabled(config.name)
        self.metrics_sink = metrics_sink
        self.replay_dir = str(self.run_dir / "replays") if (config.evaluation.save_replays or watch) else None
        self.specs = {c.name: build_spec(c) for c in config.agents}
        self.cfg_by_name = {c.name: c for c in config.agents}
        self.aug = AugmentationRegistry.from_config(config.augmentations) if config.augmentations else None
        self.guard = Guardrails(**config.guardrails.model_dump())
        self.loggers: dict[str, TrajectoryLogger] = {}
        self.results = []
        self.totals: dict[str, dict] = defaultdict(
            lambda: {"wins": 0, "games": 0, "decisions": 0, "fallbacks": 0,
                     "tokens": 0, "cost_usd": 0.0, "tool_calls": 0})
        self.extra_totals: dict[str, dict] = defaultdict(dict)  # agent extra_metrics()

    # --- context per agent (fresh agent each match, shared logger/client) ---
    def _ctx(self, name: str) -> AgentContext:
        c = self.cfg_by_name[name]
        if is_llm_type(c.type):
            logger = self.loggers.setdefault(
                name, TrajectoryLogger(self.run_dir / f"traj_{name}.jsonl", name,
                                       on_turn=self._on_turn))
            return AgentContext(
                rng=random.Random(self.cfg.seed),
                llm=self.client,
                augmentations=self.aug,
                renderer=DefaultRenderer(),
                full_info=self.cfg.full_info,
                logger=logger,
            )
        return AgentContext(rng=random.Random(self.cfg.seed))

    def _is_external(self, name: str) -> bool:
        return self.cfg_by_name[name].kind == "external"

    def _ext_logger(self, name: str) -> TrajectoryLogger:
        """Grey-box move logger for an external opponent (same JSONL shape)."""
        return self.loggers.setdefault(
            name, TrajectoryLogger(self.run_dir / f"traj_{name}.jsonl", name,
                                   on_turn=self._on_turn))

    def _pairings(self) -> list[tuple[str, str]]:
        names = [c.name for c in self.cfg.agents]
        if self.cfg.evaluation.mode == "anchored":
            anchors = self.cfg.evaluation.anchors
            return [(n, a) for n in names for a in anchors if n != a]
        return list(itertools.combinations(names, 2))  # round-robin

    def _account(self, res) -> None:
        self.totals[res.a]["wins"] += res.a_wins
        self.totals[res.b]["wins"] += res.b_wins
        self.totals[res.a]["games"] += res.n_battles
        self.totals[res.b]["games"] += res.n_battles
        for who, agent in ((res.a, getattr(res, "agent_a", None)), (res.b, getattr(res, "agent_b", None))):
            t = self.totals[who]
            t["decisions"] += getattr(agent, "n_decisions", 0)
            t["fallbacks"] += getattr(agent, "n_fallbacks", 0)
            t["tool_calls"] += getattr(agent, "n_tool_calls", 0)
            tin = getattr(agent, "tokens_in", 0) + getattr(agent, "tokens_out", 0)
            t["tokens"] += tin
            t["cost_usd"] += getattr(agent, "cost_usd", 0.0)
            self.guard.record(getattr(agent, "cost_usd", 0.0), tin)
            extra_fn = getattr(agent, "extra_metrics", None)
            if callable(extra_fn):
                try:
                    merge_extra_metrics(self.extra_totals[who], extra_fn())
                except Exception:  # noqa: BLE001  (metrics must never abort a run)
                    pass

    def _on_turn(self, rec: dict) -> None:
        """Tee from TrajectoryLogger -> per-turn TURN event (raw output truncated)."""
        raw = rec.get("output") or ""
        fields = dict(
            agent=rec.get("agent"), battle_tag=rec.get("battle_tag"),
            turn=rec.get("turn"), parsed_order=rec.get("parsed_order"),
            fallback=rec.get("fallback"), n_tool_calls=rec.get("n_tool_calls", 0),
            raw_output=raw[:2000], note=rec.get("note", ""),
        )
        extra = _scalar_extra(rec.get("extra"))
        if extra:
            fields["extra"] = extra
        self.emitter.emit(EventType.TURN, **fields)

    def _rolling_metrics(self) -> dict:
        elo = elo_from_results(self.results)
        win_rate = {n: (t["wins"] / t["games"] if t["games"] else 0.0)
                    for n, t in self.totals.items()}
        cost = {n: round(t["cost_usd"], 4) for n, t in self.totals.items()}
        return {"elo": {k: round(v, 1) for k, v in elo.items()},
                "win_rate": {k: round(v, 4) for k, v in win_rate.items()},
                "cost_usd": cost,
                "completed_pairings": len(self.results),
                "total_pairings": len(self._pairings())}

    async def run(self) -> dict:
        if self.guard.dry_run:
            return self._dry_run_report()

        tc = self.cfg.teams
        pool = DirTeamPool(tc.pool) if tc.pool else None
        tm = TeamManager(pool=pool, rng=random.Random(self.cfg.seed))
        runner = MatchRunner(team_manager=tm, rng=random.Random(self.cfg.seed))
        mode = AssignmentMode(self.cfg.teams.mode)
        teams = tm.pool.get(self.cfg.format) if (mode == AssignmentMode.MIRROR and tc.per_team > 0) else []
        blocks = team_blocks(self.cfg.evaluation.n_battles, tc.per_team if teams else 0, tc.offset, len(teams))
        if mode == AssignmentMode.POOLS:
            if not (tc.pool and tc.opponent_pool):
                raise ValueError("teams.mode pools needs teams.pool and teams.opponent_pool")
            ours, theirs = DirTeamPool(tc.pool).get(self.cfg.format), DirTeamPool(tc.opponent_pool).get(self.cfg.format)
            blocks = pool_blocks(self.cfg.evaluation.n_battles, tc.per_team, tc.offset, len(ours), len(theirs))
        pairings = self._pairings()
        self.emitter.emit(
            EventType.RUN_STARTED, agents=[c.name for c in self.cfg.agents],
            format=self.cfg.format, mode=self.cfg.evaluation.mode,
            n_pairings=len(pairings), n_battles=self.cfg.evaluation.n_battles,
            total_battles=len(pairings) * self.cfg.evaluation.n_battles)
        aborted = None
        try:
            for idx, (a, b) in enumerate(pairings):
                try:
                    self.guard.check()
                except BudgetExceeded as e:
                    aborted = str(e)
                    break
                self.emitter.emit(EventType.PAIRING_STARTED, a=a, b=b, index=idx)
                ext_name = a if self._is_external(a) else (b if self._is_external(b) else None)
                ext_logger = self._ext_logger(ext_name) if ext_name else None
                if ext_name:
                    self.emitter.emit(EventType.EXTERNAL_STARTED, name=ext_name,
                                      username=self.cfg_by_name[ext_name].external.username)
                for t_idx, nb in blocks:          # one match per mirror team (a single block by default)
                    if mode == AssignmentMode.POOLS:   # (our team, their team); the first agent plays ``pool``
                        team = (ours[t_idx[0]], theirs[t_idx[1]])
                    else:
                        team = teams[t_idx] if t_idx >= 0 else None
                    try:
                        res = await runner.run_match(
                            self.specs[a], self.specs[b], self.cfg.format, mode,
                            nb, self._ctx(a), self._ctx(b),
                            save_replays=self.replay_dir, announce=self.watch,
                            turn_delay=self.turn_delay, run_dir=str(self.run_dir),
                            ext_logger=ext_logger, team=team)
                    except Exception as e:  # noqa: BLE001  (a dead external aborts only this match)
                        if ext_name is None:
                            raise
                        self.emitter.emit(EventType.EXTERNAL_DIED, name=ext_name,
                                          message=f"{type(e).__name__}: {e}")
                        continue
                    res.team_index = t_idx[0] if isinstance(t_idx, tuple) else t_idx
                    res.opp_team_index = t_idx[1] if isinstance(t_idx, tuple) else None
                    self.results.append(res)
                    self._account(res)
                    for _ in range(res.n_battles):
                        self.guard.battle_done()
                    self.emitter.emit(
                        EventType.MATCH_FINISHED, a=res.a, b=res.b, a_wins=res.a_wins,
                        b_wins=res.b_wins, ties=res.ties, avg_turns=round(res.avg_turns, 1),
                        totals={k: dict(v) for k, v in self.totals.items()})
                    self.emitter.emit(EventType.METRIC, **self._rolling_metrics())
        except Exception as e:  # noqa: BLE001
            aborted = f"{type(e).__name__}: {e}"
            self.emitter.emit(EventType.ERROR, message=aborted)

        for lg in self.loggers.values():
            lg.close()
        report = self._report(aborted)
        report_path = self.run_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        charts_path = None
        if self.metrics_sink is not None:
            try:
                charts_path = self.metrics_sink.publish(report, self.results)
            except Exception:  # noqa: BLE001
                charts_path = None
        self.emitter.emit(EventType.RUN_FINISHED, report_path=str(report_path),
                          charts_path=str(charts_path) if charts_path else None, aborted=aborted)
        self.emitter.close()
        return report

    def _report(self, aborted: Optional[str]) -> dict:
        matrix: dict = defaultdict(dict)
        for r in self.results:                  # a pairing can be several matches (one per mirror team)
            for x, y, w, l in ((r.a, r.b, r.a_wins, r.b_wins), (r.b, r.a, r.b_wins, r.a_wins)):
                cell = matrix[x].setdefault(y, {"wins": 0, "losses": 0, "n": 0})
                cell["wins"] += w
                cell["losses"] += l
                cell["n"] += r.n_battles

        elo = elo_from_results(self.results)
        agents = {}
        for name, t in self.totals.items():
            lo, hi = wilson_interval(t["wins"], t["games"])
            agents[name] = {
                "win_rate": round(t["wins"] / t["games"], 4) if t["games"] else 0.0,
                "wilson_ci": [round(lo, 4), round(hi, 4)],
                "wins": t["wins"], "games": t["games"],
                "elo": round(elo.get(name, 1000.0), 1),
                "fallback_rate": round(t["fallbacks"] / t["decisions"], 4) if t["decisions"] else 0.0,
                "decisions": t["decisions"], "tool_calls": t["tool_calls"],
                "tokens": t["tokens"], "cost_usd": round(t["cost_usd"], 4),
            }
            if self.extra_totals.get(name):
                agents[name]["extra"] = dict(self.extra_totals[name])
        return {
            "experiment": self.cfg.name,
            "format": self.cfg.format,
            "mode": self.cfg.evaluation.mode,
            "n_battles": self.cfg.evaluation.n_battles,
            "agents": agents,
            "matrix": {k: dict(v) for k, v in matrix.items()},
            "matches": [{"a": r.a, "b": r.b, "team": getattr(r, "team_index", -1),
                         "opp_team": getattr(r, "opp_team_index", None), "a_wins": r.a_wins,
                         "b_wins": r.b_wins, "ties": r.ties, "n": r.n_battles, "avg_turns": round(r.avg_turns, 1)}
                        for r in self.results],
            "total_cost_usd": round(self.guard.spent_usd, 4),
            "total_tokens": self.guard.spent_tokens,
            "aborted": aborted,
        }

    def _dry_run_report(self) -> dict:
        n_pairs = len(self._pairings())
        # crude defaults; refine with empirical per-model numbers
        est = Guardrails.estimate(
            n_matchups=n_pairs, n_battles=self.cfg.evaluation.n_battles,
            avg_turns=30, calls_per_turn=1.5, tokens_per_call=1500,
            price_in=0.27, price_out=1.10)
        return {"experiment": self.cfg.name, "dry_run": True, "estimate": est}
