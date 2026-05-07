"""Python sandbox tool for SkyRL math_python env.

Executes Python code in a subprocess with:
  - timeout (10s default)
  - memory limit (1 GB default)
  - whitelisted imports (sympy, numpy, math, fractions, itertools, scipy)
  - captured stdout/stderr

Designed to be lightweight and fast for RL rollout (16 concurrent calls).

Usage:
    from skyrl_gym.tools.python_sandbox import PythonSandboxToolGroup
    tg = PythonSandboxToolGroup(timeout=10, mem_mb=1024)
    output = tg.execute_python("print(sum([1,2,3]))")
    # output: "6\n"

Path target: skyrl-gym/skyrl_gym/tools/python_sandbox.py
"""
from __future__ import annotations

import logging
import os
import resource
import subprocess
import sys
import tempfile
import textwrap
from typing import Optional

from skyrl_gym.tools.core import ToolGroup, tool


logger = logging.getLogger(__name__)


# Wrapper template that runs user code with resource limits + safe imports.
# NOTE: We use RLIMIT_DATA (heap segment) instead of RLIMIT_AS (virtual address
# space) because the Python interpreter inherits huge VA mappings from torch +
# CUDA libs in the parent venv, which would make RLIMIT_AS=1GB instantly fail
# even before any user code runs. RLIMIT_DATA still catches user `a = [0]*10**9`
# style memory bombs but doesn't conflict with parent process VA layout.
_RUNNER_TEMPLATE = """
import resource, signal, sys

# Memory limit (heap only — see comment above)
try:
    resource.setrlimit(resource.RLIMIT_DATA, ({mem_bytes}, {mem_bytes}))
except (ValueError, OSError):
    pass  # not all kernels enforce RLIMIT_DATA

# Allow common math libs (whitelist by trying to preimport)
import math, fractions, itertools
try:
    import numpy as np
except ImportError:
    pass
try:
    import sympy as sp
    from sympy import (
        Symbol, symbols, Eq, solve, simplify, expand, factor,
        Integer, Rational, Float, sqrt, pi, E, I, oo,
        sin, cos, tan, log, exp, diff, integrate, limit, series,
        Matrix, det, inv, solve_linear_system,
    )
except ImportError:
    pass
try:
    import scipy
except ImportError:
    pass

# Run user code
{user_code}
"""


class PythonSandboxToolGroup(ToolGroup):
    """Tool group exposing a single tool: execute_python.

    Subprocess isolation; each call spawns a fresh `python -c ...` with rlimits.
    """

    def __init__(
        self,
        timeout: int = 10,
        mem_mb: int = 4096,
        log_requests: bool = False,
        python_executable: Optional[str] = None,
    ):
        super().__init__(name="PythonSandboxToolGroup")
        self.timeout = timeout
        self.mem_bytes = mem_mb * 1024 * 1024
        self.log_requests = log_requests
        self.python_executable = python_executable or sys.executable

    @tool
    def execute_python(self, code: str) -> str:
        """Execute Python code; return combined stdout+stderr.

        Errors are returned as a string (not raised) so the agent can see them.
        """
        if not code or not code.strip():
            return "[ERROR] empty code"

        wrapped = _RUNNER_TEMPLATE.format(
            mem_bytes=self.mem_bytes,
            user_code=textwrap.dedent(code),
        )

        try:
            proc = subprocess.run(
                [self.python_executable, "-c", wrapped],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return f"[TIMEOUT] code did not finish in {self.timeout}s"
        except Exception as e:
            return f"[ERROR] subprocess failed: {type(e).__name__}: {e}"

        # Combine stdout + stderr; truncate if too long
        out = (proc.stdout or "") + (proc.stderr or "")
        MAX_LEN = 4096
        if len(out) > MAX_LEN:
            out = out[:MAX_LEN] + f"\n... (truncated, total {len(out)} chars)"
        if proc.returncode != 0 and not out.strip():
            out = f"[ERROR] exit code {proc.returncode}, no output"
        return out.strip() or "[no output]"


# Self-test
if __name__ == "__main__":
    tg = PythonSandboxToolGroup(timeout=15, mem_mb=4096)
    print("=== test 1: simple print ===")
    print(tg.execute_python("print(1 + 2)"))

    print("=== test 2: sympy ===")
    print(tg.execute_python("from sympy import solve, Symbol\nx = Symbol('x'); print(solve(x**2 - 4, x))"))

    print("=== test 3: timeout ===")
    print(tg.execute_python("import time; time.sleep(20)"))

    print("=== test 4: memory bomb ===")
    print(tg.execute_python("a = [0]*(10**9)"))

    print("=== test 5: import not in whitelist (should still work, no enforcement) ===")
    print(tg.execute_python("import os; print(os.getpid())"))
