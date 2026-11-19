"""Asymmetric knowledge-level replanning (AKR), the executor of the manuscript's Algorithm 1, as a pure-Python core.

Nothing in this package imports ROS, the simulator or the model client: the live agent (space_runtime/agent.py) and the
dry runs (experiments/akr/dryrun.py) drive the same `Executor` through the two protocols below, so every rule that the
theorems assume is exercised without a simulator. Names follow the manuscript (Definitions 2-4, eq. update, eq. adm,
eq. cover, Algorithm 1): knowledge state, record, failure set, pending cover, subgoal, episode, certified hold.

Modules (one owner each):
  types.py      shared data types (this contract; fixed)
  vocabulary.py the formal vocabulary of Definition 2 read from configs/akr_vocabulary.yaml: propositions (exogenous and
                grant subsets), actuation actions with pre/eff, sensing actions with obs/cost/window, evidence rules,
                goals per mission, the trusted initial knowledge K0 and the record R
  knowledge.py  knowledge state kappa, record R_cur, failure set F, the update operator of eq. (update) with pruning
  cover.py      exact (and greedy) solver of eq. (cover); also the relaxed problem c*_D of Theorem 2(i)
  executor.py   Algorithm 1 with the strategy switch (akr | ffreplan | kreplanner)
  accounting.py per-run summary line and the three verification checks of the execution plan, from the event log
  injection.py  fault injection and sweeps as run configuration (record, perception, proposal faults; telemetry
                removal; Earth-set time; ground decision)

Literal: `(proposition, value)` with proposition a string "name(arg,...)" and value a bool (True = the positive literal).
Knowledge value: "T" | "F" | "U". Judgment: "+" (verified) | "-" (refuted) | "?" (unverified).

World protocol (live: ROS boundary + operator console; dry run: a scripted fake world):
  world.now() -> float                                    mission time, s
  world.telemetry() -> dict[str, bool]                    true values of every exogenous proposition still reported
  world.sense(action: Action) -> dict[str, bool] | None   true values of action.obs; None = the reading is in flight
                                                          (a ground request: the reply arrives later through poll)
  world.poll() -> list[tuple[Action, dict[str, bool]]]    sensing results that arrived since the last poll
  world.execute(action: Action) -> Outcome                judgments on action.eff plus measured duration
Planner protocol:
  planner.plan(view: PlannerView) -> list[Proposal]       a finite sequence; no property is assumed (Definition 4)
  A Proposal names a capability and its arguments exactly as the model's tool call does; the vocabulary grounds it.
  A Proposal with name == REPORT is the planner's terminal report: advisory, logged, never completion.

Event log: one JSON object per line, appended through `log(kind, **fields)`; every event carries `t` (mission time).
The kinds and fields are fixed in types.EVENT_FIELDS; accounting.py reads nothing else.
"""
