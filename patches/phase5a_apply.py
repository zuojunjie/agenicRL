"""
Phase 5a ARPO patch: trajectory credit + adaptive entropy
通过 python 直接 patch（不用 patch 文件，避免格式坑）

启用方式（CLI 时设环境变量）:
  ARPO_TRAJ_CREDIT=1
  ARPO_ADAPT_ENTROPY=1
  ARPO_ENTROPY_TARGET=1.0   # default

不设环境变量 → 退化为 vanilla GRPO（向后兼容）
"""
import sys
import os

DP_ACTOR = "/root/autodl-tmp/external/Search-R1/verl/workers/actor/dp_actor.py"


def patch_dp_actor():
    src = open(DP_ACTOR).read()

    # 1. 在文件顶部 import 区后加 helper 函数 + 全局状态
    helper_code = '''

# ========== Phase 5 ARPO helpers ==========
import os as _os_arpo

class _AdaptiveEntropyState:
    """每个 actor instance 维护一个，跟随 update_policy 调用更新"""
    def __init__(self, init_coef, target=1.0, alpha=0.95):
        self.coef = init_coef
        self.ema = target
        self.target = target
        self.alpha = alpha

    def step(self, current_entropy_value):
        self.ema = self.alpha * self.ema + (1 - self.alpha) * current_entropy_value
        if self.ema < 0.5:
            self.coef = max(self.coef * 1.05, -0.5)  # explore (entropy_coef 是负的，*1.05 = 绝对值变大)
        elif self.ema > 2.5:
            self.coef = min(self.coef * 0.95, -0.0001)  # contract
        return self.coef


def _arpo_apply_traj_credit(advantages, log_prob, old_log_prob, response_mask):
    """5a Trajectory Credit Assignment: 按 |Δlog_prob| 加权 per-token advantage"""
    import torch
    log_prob_diff = (log_prob - old_log_prob).detach()  # (bs, T)
    abs_diff = log_prob_diff.abs() * response_mask
    norm = abs_diff.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    seq_len = response_mask.sum(dim=-1, keepdim=True).float().clamp_min(1.0)
    # weight: 平均值 = 1（保持 advantage 总量不变）
    weight = abs_diff / norm * seq_len
    # 只 reweight response tokens，prompt tokens 不变
    return advantages * weight * response_mask + advantages * (1 - response_mask)
# ========== Phase 5 ARPO helpers end ==========

'''

    # 找 import 区结束位置，在 from verl 那行后插入
    marker = "import verl.utils.torch_functional as verl_F"
    if marker not in src:
        print("ERROR: import marker not found")
        return False
    if "_arpo_apply_traj_credit" in src:
        print("⚠️ ARPO patch already applied, skipping helpers")
    else:
        idx = src.index(marker) + len(marker)
        # 找下一个换行
        idx = src.index("\n", idx) + 1
        src = src[:idx] + helper_code + src[idx:]
        print("✅ inserted ARPO helpers")

    # 2. 在 DataParallelPPOActor.__init__ 末尾加 adaptive entropy state init
    init_marker = "self.actor_optimizer = actor_optimizer"  # 假设这是 init 末尾的常见 line
    if "self._arpo_entropy_state" in src:
        print("⚠️ entropy state init already present")
    else:
        # 用更稳的 marker：def __init__ 的下一个 def
        # 简单方法：找 "def update_policy" 之前最后一个 self. 赋值
        if "def update_policy" not in src:
            print("ERROR: no update_policy found")
            return False
        # 找 update_policy 前 200 字符内最后一个 self. = 行
        upd_idx = src.index("def update_policy")
        before = src[max(0, upd_idx-3000):upd_idx]
        # 找最后一行 "    self.gradient_accumulation = ..."
        marker2 = "self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size"
        if marker2 not in src:
            print("ERROR: gradient_accumulation marker not found")
            return False
        idx = src.index(marker2) + len(marker2)
        idx = src.index("\n", idx) + 1
        init_addon = '''
        # ARPO 5a: adaptive entropy state（每实例一个）
        if _os_arpo.environ.get("ARPO_ADAPT_ENTROPY") == "1":
            self._arpo_entropy_state = _AdaptiveEntropyState(
                init_coef=self.config.entropy_coeff,
                target=float(_os_arpo.environ.get("ARPO_ENTROPY_TARGET", "1.0")),
            )
            print(f"[ARPO 5a] adaptive entropy enabled, target={self._arpo_entropy_state.target}")
'''
        src = src[:idx] + init_addon + src[idx:]
        print("✅ inserted entropy state init")

    # 3. 在 update_policy 里 hook traj credit (advantages 之后，compute_policy_loss 之前)
    if "_arpo_apply_traj_credit(advantages" in src:
        print("⚠️ traj credit hook already present")
    else:
        target = "advantages = data['advantages']"
        if target not in src:
            print("ERROR: advantages marker not found")
            return False
        addon = '''
                # ARPO 5a: trajectory credit assignment (per-token weight by |Δlog_prob|)
                if _os_arpo.environ.get("ARPO_TRAJ_CREDIT") == "1":
                    # log_prob 在下面才 forward；这里先 placeholder，hook 移到 forward 后'''
        # 不在这里改，移到 log_prob 后
        # 改：在 entropy, log_prob = self._forward_micro_batch(...) 之后插入 traj credit
        marker3 = "entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)"
        if marker3 not in src:
            print("ERROR: forward marker not found")
            return False
        idx = src.index(marker3) + len(marker3)
        idx = src.index("\n", idx) + 1
        traj_credit_hook = '''
                # ARPO 5a: trajectory credit (per-token weight by |Δlog_prob|)
                if _os_arpo.environ.get("ARPO_TRAJ_CREDIT") == "1":
                    advantages = _arpo_apply_traj_credit(advantages, log_prob, old_log_prob, response_mask)
'''
        src = src[:idx] + traj_credit_hook + src[idx:]
        print("✅ inserted traj credit hook")

    # 4. hook adaptive entropy（替换 entropy_coeff）
    if "_arpo_entropy_state.step" in src:
        print("⚠️ adapt entropy hook already present")
    else:
        # 在 entropy_loss 计算后、policy_loss 之前换 entropy_coeff
        marker4 = "entropy_loss = verl_F.masked_mean(entropy, response_mask)"
        if marker4 not in src:
            print("ERROR: entropy_loss marker not found")
            return False
        idx = src.index(marker4) + len(marker4)
        idx = src.index("\n", idx) + 1
        adapt_hook = '''
                # ARPO 5a: adaptive entropy (override entropy_coeff)
                if _os_arpo.environ.get("ARPO_ADAPT_ENTROPY") == "1" and hasattr(self, "_arpo_entropy_state"):
                    entropy_coeff = self._arpo_entropy_state.step(entropy_loss.detach().item())
                    metrics.setdefault("actor/arpo_entropy_coef", entropy_coeff)
                    metrics.setdefault("actor/arpo_entropy_ema", self._arpo_entropy_state.ema)
'''
        src = src[:idx] + adapt_hook + src[idx:]
        print("✅ inserted adaptive entropy hook")

    open(DP_ACTOR, "w").write(src)
    print("\n=== verify ===")
    src2 = open(DP_ACTOR).read()
    print(f"  ARPO_TRAJ_CREDIT mentions: {src2.count('ARPO_TRAJ_CREDIT')}")
    print(f"  ARPO_ADAPT_ENTROPY mentions: {src2.count('ARPO_ADAPT_ENTROPY')}")
    print(f"  _arpo_entropy_state mentions: {src2.count('_arpo_entropy_state')}")
    return True


if __name__ == "__main__":
    ok = patch_dp_actor()
    sys.exit(0 if ok else 1)
