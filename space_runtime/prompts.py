"""Where the texts the planner reads live: prompts/akr/ under the AKR executor (AKR_STRATEGY set), prompts/agent/
otherwise; a file the AKR texts do not have is read from prompts/agent/."""
import os
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def prompt(name: str) -> Path:
    if os.environ.get("AKR_STRATEGY") and (PROMPTS / "akr" / name).exists():
        return PROMPTS / "akr" / name
    return PROMPTS / "agent" / name
