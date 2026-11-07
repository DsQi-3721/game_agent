"""Weights over Foul Play's candidate sets from our opponent model's beliefs (pure Python: the look-ahead service
imports it inside Foul Play's environment, the offline tests import it in ours)."""


def belief_weights(cands, bel, rounds=4, floor=0.02):
    """Weights over candidate sets [(item, ability, tera, moves)] whose item / ability / Tera marginals match our
    beliefs restricted to the candidates' support (iterative proportional fitting from uniform), then tilted by
    the probability of each move. A value our model rules out keeps a small floor: the candidates are Foul Play's,
    the beliefs may be wrong."""
    w = [1.0] * len(cands)
    for _ in range(rounds):
        for idx, key in ((0, "items"), (1, "abilities"), (2, "tera")):
            target = bel.get(key) or {}
            if not target:
                continue
            cur = {}
            for wi, c in zip(w, cands):
                cur[c[idx]] = cur.get(c[idx], 0.0) + wi
            z = sum(cur.values()) or 1.0
            tgt = {v: target.get(v or "", 0.0) for v in cur}
            tz = sum(tgt.values())
            if tz <= 0:
                continue
            k = len(cur)
            ratio = {v: max(tgt[v] / tz, floor / k) / (cur[v] / z) for v in cur if cur[v] > 0}
            w = [wi * ratio.get(c[idx], 1.0) for wi, c in zip(w, cands)]
    q = bel.get("moves") or {}
    if q:
        w = [wi * _move_tilt(c[3], q) for wi, c in zip(w, cands)]
    return w


def _move_tilt(moves, q):
    t = 1.0
    for m in moves:
        t *= (0.05 + q.get(m, 0.0)) ** 0.5
    return t
