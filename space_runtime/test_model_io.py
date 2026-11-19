"""The planner's model_io record carries the model asked for and the model the server says answered (resp.model).

    python3 -m pytest space_runtime/test_model_io.py
"""
import json
import types

from space_runtime.tools import HarnessToolClient


def test_served_model_logged(tmp_path):
    c = HarnessToolClient.__new__(HarnessToolClient)
    c.model_id, c.step, c._model_io_path = "DeepSeek-V4.1-Flash-1017", 3, tmp_path / "model_io.jsonl"
    msg = types.SimpleNamespace(content="", reasoning_content="", tool_calls=[])
    usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15, prompt_tokens_details=None,
                                  completion_tokens_details=None)
    resp = types.SimpleNamespace(model="deepseek-v4.1-flash-1017-served", usage=usage,
                                 choices=[types.SimpleNamespace(message=msg, finish_reason="stop")])
    c._log_model_io(messages=[], resp=resp, latency_s=1.0)
    rec = json.loads(c._model_io_path.read_text())
    assert rec["model"] == "DeepSeek-V4.1-Flash-1017" and rec["served_model"] == "deepseek-v4.1-flash-1017-served"
