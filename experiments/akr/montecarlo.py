"""Monte Carlo trials at mission level (manuscript Sec. 4.7): the executor of the rover with the skill layer and the
simulator replaced by the sampled world of site_world.py, and the breadth-first planner of bfs_planner.py.

A trial draws one of the sixteen missions (in turn), a record (each entry false with probability p_R: its recorded
location replaced by another place support), the skill outcomes (rho, q), the observations (each wrong with
probability eta) and the planner's faults (with probability eps per call the plan is replaced by a faulty one: its
first irreversible action moved to the front, or the sensing before it dropped). One JSON line per trial.

    python3 -m experiments.akr.montecarlo --out <file.jsonl> --trials 1000 --grid v1 [--jobs 8]
"""
import argparse
import hashlib
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

import yaml

from space_runtime.akr.executor import Executor
from space_runtime.akr.types import IRREVERSIBLE, SENSING
from space_runtime.akr.vocabulary import Vocabulary, load_config

from experiments.akr.bfs_planner import BfsPlanner, to_proposal
from experiments.akr.plan_length import initial
from experiments.akr.site_world import SiteWorld

CFG = load_config()
MISSIONS = list(CFG["missions"])
REPAIR = {"a_l2_radiator_cover_pdu1", "a_l2_radiator_cover_bess", "a_l3_module_failure_bess", "a_l3_module_failure_isru",
          "a_l4_cannibalize_pdu1", "a_compound_fault_bess"}
RULES = ("akr", "kreplanner", "ffreplan")
# provisional until the campaign measures them (rerun at the end with the measured values); the cap L and the skill
# budget are the rover's own (configs/space_runtime_akr.yaml)
_AKR = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/space_runtime_akr.yaml").read_text())
RHO, Q, L, BUDGET = 0.95, 0.9, _AKR["akr"]["cap_L"], _AKR["reasoner"]["max_steps"]
GRIDS = {"v1": {"eps": [0.0, 0.1, 0.3, 0.6, 1.0], "p_R": [0.0, 0.1, 0.3, 0.6, 1.0], "eta": [0.0, 0.01, 0.03, 0.1]}}
# round 2 (2026-11-12, after the campaign): eps and p_R 0..1 in steps of 0.1, eps swept at the nominal p_R = 0.1 and p_R at
# eps = 0; eta swept alone; rho and q are the campaign's measured values (--rho, --q)
V2 = {"eps": [round(0.1 * k, 1) for k in range(11)], "p_R": [round(0.1 * k, 1) for k in range(11)],
      "eta": [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1], "p_R_nominal": 0.1}


class FaultyPlanner:
    """With probability eps per call the inner plan is replaced by a faulty one (Sec. 4.7)."""

    def __init__(self, inner, eps: float, rng: random.Random):
        self.inner, self.eps, self.rng, self.faults = inner, eps, rng, 0
        self.last_prompt_tokens = self.last_planner_s = None

    def plan(self, view):
        seq = self.inner.plan(view)
        acts = self.inner.last_plan
        i = next((k for k, a in enumerate(acts) if a.kind == IRREVERSIBLE), None)
        if self.eps and i is not None and self.rng.random() < self.eps:
            self.faults += 1
            if self.rng.random() < 0.5 or i == 0:          # the irreversible action moved to the front
                acts = [acts[i]] + acts[:i] + acts[i + 1:]
            else:                                           # the sensing before it dropped
                acts = [a for a in acts[:i] if a.kind != SENSING] + acts[i:]
            seq = [to_proposal(self.inner.vocab, a) for a in acts]
        return seq


def stable_seed(*parts) -> int:
    """A trial's seed from its job specification alone, the same in every process (Python's hash() of a string is
    randomized per process, so a grid generated twice would otherwise get different seeds)."""
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:4], "big")


def oracle_remedy(s: dict, vocab) -> list:
    """Planner-side oracle for the six hidden-cause repair missions: the remedy literals a
    correctly diagnosing planner would aim for, from the scenario's true causes -- the faulted unit's closed shutter
    opened, and each unserviceable module in its bays replaced by a serviceable item of the same kind (a spare not in a
    bay; with none, the module of another unit's bay, PDU-3's as in the full-stack runs). Given to the stand-in planner
    only; the executor's goal and knowledge never contain them."""
    truth, out = s["truth"], []
    units = sorted({p[len("breaker_closed("):-1] for p, _ in vocab.goal() if p.startswith("breaker_closed(")})
    bad = set(truth["unserviceable"])
    in_bay = {k: x for x, k in truth["where"].items() if k in vocab.bays}
    for u in units:
        if u in truth["drawn"]:
            out.append((f"shutter_open({u})", True))
        for bay in sorted(b for b in vocab.bays if vocab.pts[b].get("unit") == u):
            x = in_bay.get(bay)
            if x is None or x not in bad:
                continue
            same = lambda y: vocab.kind.get(y) == vocab.kind.get(x) and y not in bad and y != x
            spares = sorted(y for y, k in truth["where"].items() if same(y) and k not in vocab.bays)
            donors = sorted((y for y, k in truth["where"].items() if same(y) and k in vocab.bays and vocab.pts[k].get("unit") != u),
                            key=lambda y: (vocab.pts[truth["where"][y]].get("unit") != "pdu_3", y))
            repl = (spares or donors or [None])[0]
            if repl:
                out.append((f"item_at({repl},{bay})", True))
    return out


def falsify(manifest: dict, p_R: float, rng: random.Random, supports: list) -> tuple[dict, int]:
    m = json.loads(json.dumps(manifest))
    n = 0
    for r in m["inventory"]:
        if r["recorded_location"] and rng.random() < p_R:
            r["recorded_location"] = rng.choice([k for k in supports if k != r["recorded_location"]])
            n += 1
    return m, n


def trial(job: dict) -> dict:
    mission, rule, eps, p_R, eta, seed = (job[k] for k in ("mission", "rule", "eps", "p_R", "eta", "seed"))
    rng = random.Random(seed)
    s = initial(mission)
    v0 = Vocabulary(CFG, s["site"], s["telemetry"], s["manifest"], mission)
    manifest, false = falsify(s["manifest"], p_R, rng, v0.supports)
    vocab = Vocabulary(CFG, s["site"], s["telemetry"], manifest, mission,
                       t_vis=CFG["earth_set_s"].get(mission, CFG["earth_set_s"]["nominal"]))
    world = SiteWorld(vocab, s["telemetry"], s["truth"], rho=job.get("rho", RHO), q=job.get("q", Q), eta=eta, p_G=job.get("p_G", 0.0), seed=seed)
    planner = FaultyPlanner(BfsPlanner(vocab, limit=300_000, rule=rule), eps, rng)
    if mission in REPAIR:
        planner.inner.extra_goal = oracle_remedy(s, vocab)
    ev = []
    ex = Executor(vocab, world, planner, rule, job.get("L", L), 200, vocab.goal(), list(vocab.k0().items()), vocab.record(),
                  lambda kind, **f: ev.append({"kind": kind, **f}), mission=mission, skill_budget=BUDGET)
    planner.inner.failed = lambda: ex.K.F
    t0 = time.time()
    res = ex.run()
    disp = [e for e in ev if e["kind"] == "dispatch"]
    grants = {a.name: a.grant for a in vocab.effect_actions() if a.grant}
    return dict(job, outcome=res["outcome"], reason=res["reason"], pending=res["pending_action"], false_records=false,
                violations=sum(1 for x in world.truth_log if x["kind"] == IRREVERSIBLE and not x["pre_holds"]),
                irreversible=sum(1 for x in world.truth_log if x["kind"] == IRREVERSIBLE),
                ungranted=sum(1 for d in disp if d["action"] in grants and d["kappa_pre"].get(grants[d["action"]]) != "T"),
                refused_removals=sum(1 for x in world.truth_log if any(k in vocab.grants and not ok for k, ok in x["pre_truth"].items())),
                wrong_observations=len(world.wrong), faulty_plans=planner.faults,
                sensing_actions=sum(1 for e in ev if e["kind"] == "sense_started"),
                sensing_s=round(sum(e["cost"] for e in ev if e["kind"] == "sense_started"), 1),
                episodes=sum(1 for e in ev if e["kind"] == "episode_end"), exogenous=sum(1 for e in ev if e["kind"] == "telemetry" and e["changed_decided"]),
                planner_calls=ex.calls, skills=ex.skills, longest_plan=max(planner.inner.lengths, default=0),
                no_plan_calls=sum(1 for n in planner.inner.lengths if n == 0),
                capped_episodes=sum(1 for e in ev if e["kind"] == "episode_end" and e["reason"] == "cap"), mission_s=round(world.t, 1), wall_s=round(time.time() - t0, 2))


def jobs(grid: dict, trials: int, rules=RULES) -> list[dict]:
    """Each parameter varied alone from the all-zero point; trials spread over the sixteen missions in turn."""
    points = {(0.0, 0.0, 0.0)}
    points |= {(e, 0.0, 0.0) for e in grid["eps"]} | {(0.0, p, 0.0) for p in grid["p_R"]} | {(0.0, 0.0, h) for h in grid["eta"]}
    out = []
    for rule in rules:
        for eps, p_R, eta in sorted(points):
            for k in range(trials):
                out.append(dict(mission=MISSIONS[k % len(MISSIONS)], rule=rule, eps=eps, p_R=p_R, eta=eta,
                                seed=stable_seed(rule, eps, p_R, eta, k)))
    return out


def jobs_v2(trials: int, rho: float, q: float, rules=RULES) -> list[dict]:
    points = {(e, V2["p_R_nominal"], 0.0) for e in V2["eps"]} | {(0.0, p, 0.0) for p in V2["p_R"]} | {(0.0, 0.0, h) for h in V2["eta"]}
    return [dict(mission=MISSIONS[k % len(MISSIONS)], rule=rule, eps=eps, p_R=p_R, eta=eta, rho=rho, q=q,
                 seed=stable_seed(rule, eps, p_R, eta, k, "v2"))
            for rule in rules for eps, p_R, eta in sorted(points) for k in range(trials)]


def jobs_episodes(trials: int, rho: float, q: float, rules=("akr",)) -> list[dict]:
    """Episodes to termination at eps = p_R = eta = 0 on the ten solvable missions, at the measured rho and at rho / 2."""
    solvable = [m for m in MISSIONS if m not in REPAIR]
    return [dict(mission=solvable[k % len(solvable)], rule=rule, eps=0.0, p_R=0.0, eta=0.0, rho=r, q=q,
                 seed=stable_seed(rule, r, k, "episodes"))
            for rule in rules for r in (rho, round(rho / 2, 4)) for k in range(trials)]


def jobs_ground(trials: int, rho: float, q: float, p_G: float, rules=RULES) -> list[dict]:
    """Supplementary grid: the six hidden-cause repair missions with a ground that refuses each authorization with
    probability p_G, at every point of the main grid. Only the no-spare mission needs an authorization (a removal from
    an energized unit), so it gets `trials` trials per rule and point; the other five, trials / 5 each."""
    points = {(e, V2["p_R_nominal"], 0.0) for e in V2["eps"]} | {(0.0, p, 0.0) for p in V2["p_R"]} | {(0.0, 0.0, h) for h in V2["eta"]}
    per = {m: (trials if m == "a_l4_cannibalize_pdu1" else trials // 5) for m in sorted(REPAIR)}
    return [dict(mission=m, rule=rule, eps=eps, p_R=p_R, eta=eta, rho=rho, q=q, p_G=p_G,
                 seed=stable_seed(rule, eps, p_R, eta, m, k, p_G, "ground"))
            for rule in rules for eps, p_R, eta in sorted(points) for m, n in per.items() for k in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--grid", default="v1")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only-zero", action="store_true", help="only the all-zero point (smoke test)")
    ap.add_argument("--rho", type=float, help="effect realization probability (round 2: the campaign's measured value)")
    ap.add_argument("--q", type=float, help="conclusive-judgment probability (round 2: the campaign's measured value)")
    ap.add_argument("--p-g", type=float, default=0.5, help="ground refusal probability per authorization (grid 'ground')")
    a = ap.parse_args()
    js = jobs_ground(a.trials, a.rho, a.q, a.p_g) if a.grid == "ground" else jobs_v2(a.trials, a.rho, a.q) if a.grid == "v2" else jobs_episodes(a.trials, a.rho, a.q) if a.grid == "episodes" \
        else jobs(GRIDS[a.grid], a.trials)
    if a.only_zero:
        js = [j for j in js if j["eps"] == j["p_R"] == j["eta"] == 0.0]
    t0 = time.time()
    with open(a.out, "w") as f, Pool(a.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(trial, js, chunksize=4)):
            f.write(json.dumps(r) + "\n")
            if i % 500 == 0:
                f.flush()
                print(f"{i + 1}/{len(js)} trials, {time.time() - t0:.0f} s", flush=True)
    print(f"done: {len(js)} trials in {time.time() - t0:.0f} s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
