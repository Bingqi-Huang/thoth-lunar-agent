"""The simulator's clock does not advance while the vision-language model answers (camera question, held-item camera).

A stand-in simulator steps its clock on a thread unless paused, and serves `envs/pause` the way SRB's RosInterface
does (the answer carries the sim time the clock stands at); the model is a stub that takes 0.4 s of wall time per call.
The server's own methods run against that, unbound, so no ROS graph is needed beyond the message packages.

    python3 -m pytest openral_ext/capability_server/test_clock_hold.py      (with the ROS workspace sourced)
"""
import json
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import node  # noqa: E402

S = node.CapabilityServer


class Sim:
    """A clock stepping 2 ms of sim time per 1 ms of wall time, stopped while paused."""

    def __init__(self):
        self.t, self.paused, self.calls, self.stop = 0.0, False, [], False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop:
            if not self.paused:
                self.t += 0.002
            time.sleep(0.001)

    def pause(self, on):
        self.paused = on
        self.calls.append(on)
        return {"paused": on, "sim_t": round(self.t, 4), "step": 0}


class SlowVlm:
    def __init__(self, wall_s=0.4):
        self.wall_s = wall_s
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **_):
        time.sleep(self.wall_s)
        usage = types.SimpleNamespace(prompt_tokens=900, completion_tokens=40, prompt_tokens_details=None,
                                      completion_tokens_details=types.SimpleNamespace(reasoning_tokens=25))
        return types.SimpleNamespace(usage=usage, model="deepseek-v4.1-flash-1017-served",
                                     choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="Yes, it is open."))])


def server(tmp: Path, sim: Sim):
    img = tmp / "front.jpg"
    img.write_bytes(b"\xff\xd8\xff")
    fake = types.SimpleNamespace(vlm=SlowVlm(), _hold_n=0, _hold_lock=threading.Lock(), _vlm_log=tmp / "vlm_calls.jsonl",
                                 frame=lambda camera: {"path": img, "stamp": sim.t})
    fake._set_pause = sim.pause
    for name in ("clock_held", "ask_vlm", "observe"):
        setattr(fake, name, types.MethodType(getattr(S, name), fake))
    return fake


def test_observe_holds_the_clock():
    sim = Sim()
    with tempfile.TemporaryDirectory() as d:
        srv = server(Path(d), sim)
        time.sleep(0.05)
        before = sim.t
        t0 = time.time()
        out = srv.observe("front", "Is the shutter open?")
        wall = time.time() - t0
        after = sim.t
        rec = json.loads((Path(d) / "vlm_calls.jsonl").read_text().splitlines()[-1])
    sim.stop = True
    assert out["answer"].startswith("Yes") and wall >= 0.4
    assert sim.calls == [True, False]
    # held for the whole model call: the clock moved by the pause latency at most, not by 0.4 s x 2 = 0.8 s
    assert rec["sim_t_released"] - rec["sim_t_stopped"] < 0.01
    assert after - before < 0.1
    assert rec["query"] == "observe" and rec["calls"][0]["reasoning_tokens"] == 25 and rec["calls"][0]["wall_s"] >= 0.4
    # the model asked for and the model the server says answered
    assert rec["calls"][0]["model"] == node.VLM_MODEL and rec["calls"][0]["served_model"] == "deepseek-v4.1-flash-1017-served"


def test_concurrent_holds_share_one_pause():
    """detect asks several times at once: one pause for all of them, released when the last one returns."""
    sim = Sim()
    with tempfile.TemporaryDirectory() as d:
        srv = server(Path(d), sim)
        with srv.clock_held("detect") as rec:
            ts = [threading.Thread(target=srv.ask_vlm, args=(Path(d) / "front.jpg", "where?", rec)) for _ in range(4)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            inner = sim.paused
        line = json.loads((Path(d) / "vlm_calls.jsonl").read_text().splitlines()[-1])
    sim.stop = True
    assert inner and sim.calls == [True, False] and len(line["calls"]) == 4
    assert line["sim_t_released"] - line["sim_t_stopped"] < 0.01


def test_released_when_the_model_fails():
    sim = Sim()
    with tempfile.TemporaryDirectory() as d:
        srv = server(Path(d), sim)
        srv.vlm.chat.completions.create = lambda **_: (_ for _ in ()).throw(TimeoutError("gateway"))
        try:
            srv.observe("front", "Is the lamp green?")
        except TimeoutError:
            pass
    sim.stop = True
    assert sim.calls == [True, False] and not sim.paused
