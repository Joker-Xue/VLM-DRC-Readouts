"""Mechanism statistics for Section V.C, recomputed from stored debate transcripts.

Two families of statistic the paper quotes:

  A) Round-1-confident-then-collapse (A2) and round-1-divided-then-flood (A3) counts.
  B) `depends_on_agents` citation-consistency of round-2/3 confidence revisions.

Both were originally computed ad hoc; this script pins the definitions down so they can be
re-run when the headline sample size changes. Run it on the old n=451 directory first to
confirm the definitions reproduce the published numbers before trusting the n=2000 output.

Usage: PYTHONPATH=. python3 experiments/eval/debate_transcript_stats.py [dir ...]
"""
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
AGENTS = ["congestion_analyst", "wire_demand_forecaster", "drc_pattern_forensics", "layout_arbiter"]


def rounds_of(arm_obj):
    """raw_messages is a list of per-round dicts: {round, votes, messages:[...]}"""
    out = {}
    for blk in arm_obj.get("raw_messages", []):
        if isinstance(blk, dict) and "round" in blk and "messages" in blk:
            out[blk["round"]] = blk["messages"]
    return out


def conf_by_agent(msgs):
    d = {}
    for m in msgs:
        a = m.get("agent")
        c = m.get("confidence")
        if a is not None and isinstance(c, (int, float)):
            d[a] = float(c)
    return d


def stance(m):
    """hotspot-leaning? from message_type or vote."""
    if m.get("message_type") == "hotspot_proposal":
        return "hotspot"
    if m.get("message_type") == "not_hotspot":
        return "not_hotspot"
    if m.get("message_type") == "abstain":
        return "abstain"
    return m.get("vote")


def a2_collapse_stats(rows):
    """Strong-tier regions where A2 starts confident+majority-correct, then abstains out."""
    strong = [r for r in rows if r["bucket"] == "strong[>=0.20]"]
    usable, confident, collapsed = 0, 0, 0
    for r in strong:
        rd = rounds_of(r["arms"]["A2"])
        if 1 not in rd or not rd[1]:
            continue
        usable += 1
        c1 = conf_by_agent(rd[1])
        if len(c1) < 3:
            continue
        votes1 = [stance(m) for m in rd[1]]
        majority_hot = sum(v == "hotspot" for v in votes1) > len(votes1) / 2
        mean_ok = sum(c1.values()) / len(c1) >= 0.5
        floor_ok = min(c1.values()) >= 0.4
        if not (majority_hot and mean_ok and floor_ok):
            continue
        confident += 1
        last = max(rd)
        abst = sum(stance(m) == "abstain" for m in rd[last])
        if abst >= 3:
            collapsed += 1
    return usable, confident, collapsed


def a3_flood_stats(rows):
    """Clean-tier regions ending in unanimous A3 hotspot vote; how many started divided."""
    clean = [r for r in rows if r["bucket"] == "clean(<0.01)"]
    floods, divided = 0, 0
    for r in clean:
        rd = rounds_of(r["arms"]["A3"])
        if 1 not in rd or not rd:
            continue
        last = max(rd)
        votes_last = [stance(m) for m in rd[last]]
        if not votes_last or not all(v == "hotspot" for v in votes_last) or len(votes_last) < 4:
            continue
        floods += 1
        c1 = conf_by_agent(rd[1])
        if c1 and min(c1.values()) < 0.5:
            divided += 1
    return floods, divided


def citation_stats(rows, arm):
    """Do round-2/3 revisions move toward the peer(s) they cite?"""
    tot = resolvable = toward = 0
    up = down = 0
    up_ok = down_ok = 0
    for r in rows:
        rd = rounds_of(r["arms"][arm])
        for rnd in (2, 3):
            if rnd not in rd or (rnd - 1) not in rd:
                continue
            prev = conf_by_agent(rd[rnd - 1])
            for m in rd[rnd]:
                if m.get("message_type") != "vote":
                    continue
                tot += 1
                a = m.get("agent")
                cites = m.get("depends_on_agents") or []
                c_now = m.get("confidence")
                if a is None or not cites or not isinstance(c_now, (int, float)):
                    continue
                if a not in prev:
                    continue
                peer_vals = [prev[p] for p in cites if p in prev]
                if not peer_vals:
                    continue
                resolvable += 1
                c_prev = prev[a]
                delta = float(c_now) - c_prev
                if delta == 0:
                    continue
                peer_mean = sum(peer_vals) / len(peer_vals)
                moved_toward = (delta > 0 and peer_mean > c_prev) or (delta < 0 and peer_mean < c_prev)
                if delta > 0:
                    up += 1
                    up_ok += moved_toward
                else:
                    down += 1
                    down_ok += moved_toward
                toward += moved_toward
    directional = up + down
    return dict(total_votes=tot, resolvable=resolvable, directional=directional,
                toward=toward, up=up, down=down, up_ok=up_ok, down_ok=down_ok)


def report(dirname):
    rows = json.loads((RES / dirname / "per_region.json").read_text())
    print("=" * 78)
    print(f"{dirname}  n={len(rows)}")
    print("=" * 78)

    usable, confident, collapsed = a2_collapse_stats(rows)
    pct = f"{collapsed/confident:.1%}" if confident else "n/a"
    print(f"A2 deadlock: strong-tier regions with a round 1 = {usable}; "
          f"confident+majority-correct at round 1 = {confident}; "
          f"of those, ending in >=3/4 abstention = {collapsed} ({pct})")

    floods, divided = a3_flood_stats(rows)
    pct = f"{divided/floods:.1%}" if floods else "n/a"
    print(f"A3 flood: clean-tier regions ending unanimous-hotspot = {floods}; "
          f"of those, round 1 genuinely divided (min conf < 0.5) = {divided} ({pct})")

    for arm in ("A2", "A3"):
        s = citation_stats(rows, arm)
        dr = s["directional"]
        print(f"\n{arm} citation consistency: {s['resolvable']}/{s['total_votes']} votes resolvable "
              f"({s['resolvable']/s['total_votes']:.0%}); directional moves = {dr}")
        if dr:
            print(f"   toward cited peer overall: {s['toward']}/{dr} = {s['toward']/dr:.1%}")
        if s["up"]:
            print(f"   upward   {s['up']:>4}  consistent {s['up_ok']/s['up']:.1%}")
        if s["down"]:
            print(f"   downward {s['down']:>4}  consistent {s['down_ok']/s['down']:.1%}")
    print()


if __name__ == "__main__":
    for d in sys.argv[1:] or ["phase7_arms_v3gm_n450", "phase7_arms_v3gm_n2000_v2"]:
        report(d)
