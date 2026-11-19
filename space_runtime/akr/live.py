"""The AKR executor on the live graph: the World protocol over the common capability boundary and the operator console,
and the Planner protocol over the foundation model (Qwen3.8-Max through `ReasonerCore`).

The model is consulted at every decision (manuscript 4.3): one planner call runs model steps until the model proposes
a modelled action -- a capability effect, a typed sensing call (serial-plate read, held-item check, an authorization
request) -- or its terminal report, and returns that one proposal. Everything else the model does in between (its
working plan, free camera questions, site and manifest queries, information requests to mission control) is
dispatched as before and never reaches the knowledge state. `continues` tells the executor whether the working plan
was left unchanged, so that the episode goes on (manuscript 4.3: an episode is the interval during which the working
plan is unchanged).
"""
import json
import time
import uuid

from openral_core import EmitPromptTool
from openral_msgs.msg import PromptStamped
from std_srvs.srv import SetBool

from openral_ext.capability_server.contracts import CONTRACTS
from space_runtime.akr.types import REPORT, Outcome, Proposal
from space_runtime.tools import CHECK, DECOMPOSE, REPORT_TOPIC, SET_TASK_STATUS, WAIT
from space_runtime.verification import verify


class RosWorld:
    """Definition 2's world as the boundary realizes it. Every reading and every call is also appended to the run's
    event record in the kinds the context and the evaluator already read (query_answered, execution_*, effect_verified,
    operator_request_sent), so the model's context and the old reports see the same run."""

    def __init__(self, agent, vocab, knowledge):
        self.agent, self.vocab, self.knowledge = agent, vocab, knowledge   # knowledge: () -> the executor's Knowledge
        self.requests = {}                                                 # request_id -> sensing action in flight

    def now(self) -> float:
        return self.agent.sim_t

    def telemetry(self) -> dict:
        t = self.agent.query("power_status", {})
        self.agent.event("equipment_telemetry", value=t["value"] if t["success"] else None)
        return self.vocab.telemetry_values(t["value"]) if t["success"] else {}

    def _own(self, a) -> None:
        """A reading or call the executor starts itself (a pending-cover element) gets its own step entry, marked as the
        executor's: it does not overwrite the model's last step in the context and does not count against its budget."""
        recent = self.agent.store.facts["recent"]
        if not (recent and recent[-1]["tool"] in (a.capability, "request_operator") and "result" not in recent[-1]):
            self.agent.event("tool_proposed", tool=a.capability, args=dict(a.args), by="executor",
                             rationale=f"executor: sensing {', '.join(a.obs)} for its pending cover")

    def sense(self, a):
        ag, args = self.agent, dict(a.args)
        self._own(a)
        if a.capability == "request_operator":
            pending = [{"capability": args["authorize_capability"], "target": args["authorize_target"],
                        "reason": f"precondition {a.obs[0]} undecided"}]
            rid = uuid.uuid4().hex[:12]
            ag.request_pub.publish(PromptStamped(text=f"Authorization requested for {pending[0]['capability']} on {pending[0]['target']}.",
                                                 metadata_json=json.dumps({"request_id": rid, "kind": "authorization",
                                                                           "pending_approvals": pending})))
            ag.event("operator_request_sent", request_id=rid, request_kind="authorization", text="(sent by the executor)",
                     pending_approvals=pending)
            ag.open_requests.add(rid)
            self.requests[rid] = a
            return None
        if a.capability == "scan_label":
            effects = self._run(a)
            return self.vocab.read(a, effects)
        if a.capability == "held_check":
            return self._held_check(args["support"])
        q = ag.query(a.capability, args)
        ag.event("query_answered", query=a.capability, args=args, by="executor", **q)
        return self.vocab.read(a, q["value"]) if q["success"] else {}

    def _held_check(self, support: str) -> dict:
        """The held-item check: set the held item down on the free support, read its plate there, pick it up again with
        the arguments of the item's own last pick (its support and lifting height moved to this support)."""
        ag, v = self.agent, self.vocab
        last = next(e for e in reversed(ag.store.events) if e["kind"] == "execution_started" and e["capability"] == "pick")["args"]
        place = next((e["args"] for e in reversed(ag.store.events) if e["kind"] == "execution_started" and e["capability"] == "place"), {})
        z = lambda s: v.pts[s]["xyz"][2]
        pick_args = {**last, "support": support}
        if "grasp_height_m" in last:
            pick_args["grasp_height_m"] = round(last["grasp_height_m"] - z(last["support"]) + z(support), 3)
        kind = last.get("item") or v.cfg["scan"]["default_item"]
        self._run(v.ground_call("place", {"support": support, "item": kind,
                                          **({"release_constraint": place["release_constraint"]} if "release_constraint" in place else {})}))
        read = v.read(v.scan(support, kind), self._run(v.ground_call("scan_label", {"support": support, "item": kind})))
        serial = next((p[len("item_at("):].split(",")[0] for p, val in read.items() if p.startswith("item_at(") and val), None)
        picked = self._run(v.ground_call("pick", pick_args))
        repicked = any(e["effect"] == "item_held" and e["holds"] for e in picked)
        return v.read(v.held_check(support), {"serial": serial, "repicked": repicked})

    def poll(self) -> list:
        """Decisions on the executor's requests, wherever the answer was recorded (the planner also takes replies)."""
        ag = self.agent
        ag.take_operator_replies()
        out = []
        for ev in ag.store.events:
            if ev["kind"] == "operator_answered" and ev["request_id"] in self.requests and ev.get("outcome") in ("grant", "deny"):
                a = self.requests.pop(ev["request_id"])
                out.append((a, self.vocab.read(a, {"grant": ev["outcome"] == "grant"})))
        return out

    def execute(self, a) -> Outcome:
        t0 = self.agent.sim_t
        kappa_pre = {p: self.knowledge().value(p) for p, _ in a.pre}
        effects = self._run(a)
        tele = None
        if a.capability == "press":             # its equipment effects are judged by the reading after it (Sec. 4.2)
            t = self.agent.query("power_status", {})
            self.agent.event("equipment_telemetry", value=t["value"] if t["success"] else None, after=a.name)
            tele = t["value"] if t["success"] else None
        return Outcome(self.vocab.judge(a, effects, kappa_pre, tele), round(self.agent.sim_t - t0, 2),
                       {"verified": [{k: v for k, v in e.items() if k in ("effect", "holds")} for e in effects],
                        **({"telemetry_after": tele} if tele is not None else {})})

    def _run(self, a) -> list[dict]:
        """One call through the boundary with the independent verification of its effects (agent.dispatch's effect
        path). The guard keeps only its execution checks (contract arguments, availability, active faults); the
        authority rule is the grant precondition the executor checks."""
        ag, name, args = self.agent, a.capability, dict(a.args)
        outcome, reason = ag.guard.check(name, args, ag.store.facts)
        if outcome in ("deny", "unavailable"):
            ag.event("guard_decision", tool=name, outcome=outcome, reason=reason)
            return []
        ag.event("execution_started", capability=name, args=args)
        r = ag.execute(name, args)
        ag.event("execution_finished", capability=name, args=args, **r)
        held = [n for n, o in ag.store.facts["objects"].items() if o["state"] == "in_gripper"]
        look = ag.query("held_item", {}) if held and name not in ("place", "release") else None
        effects = verify(name, args, r["evidence"], ag.cfg["verification"], held, look["value"] if look and look["success"] else None)
        for effect in effects:
            ag.event("effect_verified", capability=name, **effect)
        ag.event("verification_finished", capability=name)
        return effects


#: How long a pause / resume request is left unanswered before it is sent again (it is idempotent). The simulator answers a
#: pause once the step in flight is done, which at a rendering run's 0.06x real time is a second or two.
CLOCK_HOLD_RESEND_S = 15.0


class LlmPlanner:
    """The model as the planner of Definition 4: no property of it is relied upon."""

    def __init__(self, agent, vocab, max_steps: int, guard_calls: int = 200):
        """`max_steps` is the run's skill budget the model is shown; `guard_calls` an infrastructure guard on model calls."""
        self.agent, self.vocab, self.max_steps, self.guard_calls = agent, vocab, max_steps, guard_calls
        self.continues, self.exhausted = False, False
        self.last_prompt_tokens = self.last_planner_s = None
        self.executor = None

    def steps_used(self) -> int:
        from space_runtime.agent import PLAN_TOOLS
        return sum(ev["kind"] == "reasoner_step_failed" or (ev["kind"] == "tool_proposed" and ev["tool"] not in PLAN_TOOLS
                                                              and ev.get("by") != "executor")
                   for ev in self.agent.store.events)

    def plan(self, view) -> list:
        ag = self.agent
        ag.renderer.akr = {"view": view, "executor": self.executor}
        ledger0, tokens, t0 = ag.store.facts["ledger"], 0, time.time()
        while True:
            if ag.step >= self.guard_calls:
                self.exhausted = True
                return []
            ag.take_operator_replies()
            ag.step += 1
            ag.renderer.step = (self.executor.skills if self.executor else 0) + 1   # skill calls used, as the budget counts them
            ag.client.step = ag.step
            ag.event("reasoner_step_started", step=ag.step)
            res = self._model_call()   # also sets ag._llm_s: the tick's own wall time, without the pause round trips
            tokens += ag.client.last_prompt_tokens or 0
            if res.tool_call is None:
                raw = ag.client.last_raw or {}
                ag.event("reasoner_step_failed", tool=raw.get("name", "?"), args=raw.get("arguments", ""), llm_s=ag._llm_s,
                         detail=str(res.error) if res.error else f"suppressed: {res.suppressed_reason}")
                continue
            p = self._proposal(res.tool_call)
            if p is not None:
                self.continues = ag.store.facts["ledger"] == ledger0
                self.last_prompt_tokens, self.last_planner_s = tokens, round(time.time() - t0, 1)
                return [p]

    def _hold_clock(self, on: bool) -> dict:
        """The simulator's `envs/pause` (SRB RosInterface): True stops the clock once the step in flight is done, False
        lets it continue. The answer is `{paused, sim_t, step}` -- the sim time the clock stands at. A request is
        idempotent, so one whose response is lost (Humble drops the first answer after discovery) is simply sent again."""
        ag = self.agent
        req = SetBool.Request(data=on)
        futures = [ag.pause_cli.call_async(req)]
        while not any(f.done() for f in futures):
            t = time.time()
            while not any(f.done() for f in futures) and time.time() - t < CLOCK_HOLD_RESEND_S:
                time.sleep(0.02)
            if not any(f.done() for f in futures):
                futures.append(ag.pause_cli.call_async(req))
        r = next(f for f in futures if f.done()).result()
        assert r.success, f"envs/pause(data={on}) failed: {r.message}"
        return json.loads(r.message)

    def _model_call(self):
        """One `core.tick` -- the model call -- with the simulator's clock held. Mission time is simulated time; what the
        model costs in wall seconds (its latency, the gateway's queue) is not time the mission spends, and no capability
        runs while the agent waits (every effect has completed before the next step), so the rover stands still. The
        `model_call` event records the sim time the clock stood at when it was stopped, when the call returned (still
        stopped; the agent's own /clock subscription) and when it was let go: all three are the same sim time."""
        ag = self.agent
        assert ag.pause_cli.wait_for_service(timeout_sec=60), "the simulator has no envs/pause service (an SRB build without it)"
        stopped = self._hold_clock(True)
        t0 = time.time()
        try:
            res = ag.core.tick(world_state=None, renderer=ag.renderer, palette=ag.palette, force=True)
        finally:
            wall_s, clock_at_return = time.time() - t0, ag.sim_t
            ag._llm_s = round(wall_s, 1)
            released = self._hold_clock(False)
        ag.event("model_call", step=ag.step, wall_s=round(wall_s, 2), sim_t_stopped=round(stopped["sim_t"], 3),
                 sim_t_at_return=round(clock_at_return, 3), sim_t_released=round(released["sim_t"], 3),
                 sim_step_stopped=stopped["step"], sim_step_released=released["step"])
        return res

    def _proposal(self, call):
        """The modelled proposal this tool call is, or None after dispatching it the old way (it touches no knowledge)."""
        ag = self.agent
        if isinstance(call, EmitPromptTool):
            meta = json.loads(call.metadata_json) if call.metadata_json else {}
            tool = "report_to_operator" if call.target_topic == REPORT_TOPIC else "request_operator"
            if tool == "report_to_operator":
                ag.proposed(tool=tool, args={"text": call.text}, rationale=call.rationale)
                ag.report_pub.publish(PromptStamped(text=call.text))
                return Proposal.of(REPORT, {"text": call.text})
            named = f"{self.vocab.cfg['grant']['family']}({meta.get('authorize_target')})"
            grant = named if named in self.vocab.grants else self._held_grant()
            if meta.get("kind") == "authorization" and grant:
                ag.proposed(tool=tool, args={"text": call.text, **meta}, rationale=call.rationale)
                return Proposal.of("request_operator", {"kind": "authorization", "grant": grant})
            ag.dispatch(call)
            return None
        name, args = call.rskill_id, json.loads(call.goal_params_json or "{}")
        if name == CHECK:
            ag.proposed(tool="check_proposition", args=args, rationale=call.rationale)
            return Proposal.of("check_proposition", args)
        if name in (DECOMPOSE, SET_TASK_STATUS, WAIT):
            ag.dispatch(call)
            return None
        if name in CONTRACTS["queries"] and self.vocab.ground(Proposal.of(name, args), self.executor.K) is None:
            ag.dispatch(call)
            return None
        ag.proposed(tool=name, args=args, rationale=call.rationale)
        return Proposal.of(name, args)

    def _held_grant(self) -> str | None:
        """The grant proposition of the restricted action the executor last refused for undecided preconditions."""
        ex = self.executor
        names = sorted(ex.dp_of, key=ex.dp_of.get)
        held = [a for a in self.vocab.effect_actions() if a.grant and a.name in names]
        return max(held, key=lambda a: ex.dp_of[a.name]).grant if held else None
