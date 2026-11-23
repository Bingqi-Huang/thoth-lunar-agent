"""Summary of a Monte Carlo trial file (montecarlo.py): per acting rule and grid point, the fraction of trials with a
safety violation, completion on the ten solvable missions, sensing time, episodes against the episode bound N, and
the outcome of the six repair missions.

    python3 -m experiments.akr.mc_summary <trials.jsonl>
"""
import collections
import json
import statistics as st
import sys

from experiments.akr.montecarlo import REPAIR


def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["rule"], r["eps"], r["p_R"], r["eta"])].append(r)
    print(f"{len(rows)} trials")
    print(f"{'rule':10s} {'eps':>4} {'p_R':>4} {'eta':>5} {'n':>4} {'viol':>6} {'ungr':>5} {'solv ok':>8} {'repair ok':>9} "
          f"{'sens s med':>10} {'episodes med':>12} {'no plan':>7} {'longest':>7} {'capped':>6}")
    for (rule, eps, p, eta), R in sorted(by.items()):
        solv = [r for r in R if r["mission"] not in REPAIR]
        rep = [r for r in R if r["mission"] in REPAIR]
        print(f"{rule:10s} {eps:4.2f} {p:4.2f} {eta:5.3f} {len(R):4d} {sum(r['violations'] > 0 for r in R) / len(R):6.3f} "
              f"{sum(r['ungranted'] > 0 for r in R):5d} {sum(r['outcome'] == 'success' for r in solv):4d}/{len(solv):<3d} "
              f"{sum(r['outcome'] == 'success' for r in rep):4d}/{len(rep):<4d} {st.median(r['sensing_s'] for r in R):10.1f} "
              f"{st.median(r['episodes'] for r in solv):12.1f} {sum(r['no_plan_calls'] for r in R):7d} {max(r['longest_plan'] for r in R):7d} "
              f"{sum(r['capped_episodes'] for r in R):6d}")
    print("\nnot completed among the ten solvable missions, by rule / point / ending:")
    c = collections.Counter((r["rule"], r["eps"], r["p_R"], r["eta"], r["mission"], f"{r['outcome']}/{r['reason']}")
                            for r in rows if r["mission"] not in REPAIR and r["outcome"] != "success")
    for k, n in sorted(c.items()):
        print("  ", k, n)
    print("\nrepair missions, endings:", collections.Counter(f"{r['outcome']}/{r['reason']}" for r in rows if r["mission"] in REPAIR))


if __name__ == "__main__":
    main()
