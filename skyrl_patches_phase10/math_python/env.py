"""MathPythonEnv — SkyRL env for MATH+Python (Phase 10).

Mirror of SearchEnv structure, but:
  - Tool: PythonSandboxToolGroup (not SearchToolGroup)
  - Action parser: <python>...</python> (not <search>)
  - Tool output wrap: <output>...</output> (not <information>)
  - Done condition: same (<answer>...</answer> or max_turns reached)
  - Reward: math_python.utils.compute_score (sympy verifier)

Path target: skyrl-gym/skyrl_gym/envs/math_python/env.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.math_python.utils import compute_score
from skyrl_gym.tools.python_sandbox import PythonSandboxToolGroup


@dataclass
class MathPythonEnvConfig:
    timeout: int = 10        # python sandbox timeout (sec)
    mem_mb: int = 1024       # python sandbox memory (MB)
    log_requests: bool = False


class MathPythonEnv(BaseTextEnv):
    """Environment for MATH problems with Python tool execution.

    Trajectory shape:
        <think>...</think>
        <python>code</python>
        <output>...</output>      ← injected by env
        <think>...</think>
        ...
        <answer>X</answer>        ← terminal
    """

    def __init__(
        self,
        env_config: Union[MathPythonEnvConfig, DictConfig],
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth in reward_spec required"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_turns = extras.get("max_turns", 8)

        # Phase 7.8e patch: eval mode forces binary EM (here always binary, kept for compat)
        self.is_eval = extras.get("is_eval", False)

        # Init Python sandbox tool
        timeout = getattr(env_config, "timeout", 10)
        mem_mb = getattr(env_config, "mem_mb", 1024)
        log_requests = getattr(env_config, "log_requests", False)
        self.tool_group = PythonSandboxToolGroup(
            timeout=timeout, mem_mb=mem_mb, log_requests=log_requests
        )
        self.init_tool_groups([self.tool_group])

        self.chat_history: ConversationType = []

    def _parse_action(self, action: str) -> List[Optional[str]]:
        """Extract Python code from <python>...</python> tag."""
        match = None
        if "<python>" in action and "</python>" in action:
            match = re.search(r"<python>(.*?)</python>", action, re.DOTALL)
        return [match.group(1)] if match else [None]

    def _is_done(self, action: str) -> bool:
        if self.turns >= self.max_turns:
            return True
        return "<answer>" in action and "</answer>" in action

    def _get_reward(self, action: str, done: bool) -> float:
        if not done:
            return 0.0
        chat_history_str = "".join([item["content"] for item in self.chat_history])
        return compute_score(chat_history_str, self.ground_truth)

    def _execute_tool(self, tool_group_name: str, tool_name: str, tool_input: Any) -> str:
        tool_output = super()._execute_tool(tool_group_name, tool_name, tool_input)
        # Wrap output in <output> tag (analogous to <information> in search env)
        return "\n<output>" + tool_output + "</output>\n"

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})

        error = None
        done = self._is_done(action)
        reward = self._get_reward(action, done)

        if done:
            return BaseTextEnvStepOutput(
                observations=[], reward=reward, done=done, metadata={}
            )

        try:
            code = self._parse_action(action)
            if code[0] is None:
                # No <python> tag found, no <answer> either → invalid action
                error = "[Error] No <python>...</python> or <answer>...</answer> found in response."
                observation = None
            else:
                observation = self._execute_tool(
                    "PythonSandboxToolGroup", "execute_python", code
                )
        except Exception as e:
            error = f"[Error] {type(e).__name__}: {e}"
            observation = None

        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            new_obs = {"role": "user", "content": error}
        else:
            new_obs = None

        info = {
            "tool_group": "PythonSandboxToolGroup",
            "tool_name": "execute_python",
            "tool_input": code if code is not None else None,
        }

        if new_obs:
            self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs] if new_obs else [],
            reward=reward,
            done=done,
            metadata=info,
        )
