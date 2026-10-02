"""Defaults that decide quality and spend. A deployed service has no .env file, so
these are what it runs with unless they're set explicitly."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
QUALITY = {"summary_effort": "medium", "agent_effort": "high", "summary_model": "claude-sonnet-5-5",
           "agent_model": "claude-opus-5-5", "job_runner": "thread"}


def test_defaults_without_env_or_dotenv():
    # A fresh interpreter with none of the settings in its environment and .env loading
    # switched off (config reads both at import time).
    names = {"SUMMARY_EFFORT", "AGENT_EFFORT", "SUMMARY_MODEL", "AGENT_MODEL", "JOB_RUNNER"}
    env = {k: v for k, v in os.environ.items() if k not in names}
    code = ("import json, dotenv; dotenv.load_dotenv = lambda *a, **k: False; import config; "
            f"print(json.dumps({{k: getattr(config.settings, k) for k in {sorted(QUALITY)!r}}}))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == QUALITY


def test_env_example_matches_the_defaults():
    example = dict(line.split("=", 1) for line in (ROOT / ".env.example").read_text().splitlines()
                   if "=" in line and not line.lstrip().startswith("#"))
    assert {k: example[k.upper()].split("#")[0].strip() for k in QUALITY} == QUALITY
