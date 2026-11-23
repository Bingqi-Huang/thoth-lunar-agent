"""The site survey's button descriptions (configs/button_appearance.yaml) are word for word the ones the authored
baselines' task models carry, original and revised: every method's press looks for the
same button text. Reads the files; imports nothing of the boundary.

    python3 -m pytest openral_ext/capability_server/test_button_appearance.py
"""
import glob
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
APPEARANCE = yaml.safe_load((ROOT / "configs/button_appearance.yaml").read_text())
PDU = [f"pdu_{k}_panel/press" for k in (1, 2, 3)]
SHUTTERS = [k for k in APPEARANCE if k.endswith("/cover_button")]


def one(pattern: str, text: str) -> str:
    found = set(re.findall(pattern, text))
    assert len(found) == 1, (pattern, found)
    return found.pop()


def test_every_press_point_has_a_description():
    ix = __import__("json").loads((ROOT / "space_robotics_bench/assets/lunar_base/usd/scene/lunar_base/interaction.json").read_text())
    assert {k for k, v in ix["points"].items() if v["type"] == "press"} == set(APPEARANCE)


def test_europa_hosts():
    for f in [ROOT / "baselines/europa/host.py", *map(Path, glob.glob(str(ROOT / "baselines/revised/europa-*/europa/host.py")))]:
        s = f.read_text()
        assert one(r'\{"target": "([^"]+)"\} for u in \("pdu_1", "pdu_2", "pdu_3"\)', s) == APPEARANCE[PDU[0]], f
        assert one(r'"bess": \{"target": "([^"]+)"\}', s) == APPEARANCE["bess_panel/press"], f
        assert one(r'"isru": \{"target": "([^"]+)"\}', s) == APPEARANCE["isru_panel/press"], f
        if "SHUTTER_TARGET =" in s:
            assert all(one(r'SHUTTER_TARGET = "([^"]+)"', s) == APPEARANCE[k] for k in SHUTTERS), f


def test_plexil_plans():
    for f in [ROOT / "baselines/plexil/plans/MissionA_Covered.ple", *map(Path, glob.glob(str(ROOT / "baselines/revised/plexil-*/plexil/plans/MissionA_Covered.ple")))]:
        s = f.read_text()
        assert one(r'Unit == "bess"\) \{ ButtonTarget = "([^"]+)"', s) == APPEARANCE["bess_panel/press"], f
        assert one(r'Unit == "isru"\) \{ ButtonTarget = "([^"]+)"', s) == APPEARANCE["isru_panel/press"], f
        assert one(r'else \{ ButtonTarget = "([^"]+)"', s) == APPEARANCE[PDU[0]], f
        for text in re.findall(r'press\("([^"]+)", Shutter\)', s):
            assert all(text == APPEARANCE[k] for k in SHUTTERS), f


def test_description_per_kind_is_one_text():
    assert len({APPEARANCE[k] for k in PDU}) == 1 and len({APPEARANCE[k] for k in SHUTTERS}) == 1
