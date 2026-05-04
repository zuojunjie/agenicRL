"""
Phase 5b ARPO patch: turn-level KL（叠加在 5a 上）
通过 python 直接 patch dp_actor.py

启用方式（CLI 时设环境变量）:
  ARPO_TURN_KL=1
  ARPO_BETA_THINK=0.001     # think 段（鼓励 reasoning 探索）
  ARPO_BETA_SEARCH=0.005    # search 段（抑制 query 漂移）
  ARPO_BETA_ANSWER=0.001    # answer 段（保持答题灵活）
  ARPO_BETA_INFO=0.0        # information 段（环境注入，不参与 KL）
  ARPO_BETA_OTHER=0.001     # 标签外 / 不在任何 turn 的 token

不设 → 退化 5a-only 行为（向后兼容）

设计要点:
  1. 用 tokenizer 预编码 tag 序列（init 时一次性，不动训练 path）
  2. 每 batch 扫一遍 response_ids 标记 turn type（per-sample 状态机）
  3. KL loss 按 turn type 加权 β（替换原 single-β 计算）

边界 case:
  - tokenizer 可能把 `<think>` 拆成 `<` `think` `>` 多 token
    → 用 input_ids 序列匹配（把整 tag string 编码后整体比对）
  - 状态遗留：跨 turn 切换时清零，避免 mask 串错
"""
import sys

DP_ACTOR = "/root/autodl-tmp/external/Search-R1/verl/workers/actor/dp_actor.py"


def patch_dp_actor():
    src = open(DP_ACTOR).read()

    # 0. 检查 5a 是否已 apply（5b 依赖 5a 的 _arpo_* helpers 区域）
    if "_arpo_apply_traj_credit" not in src:
        print("⚠️ 5a patch 未先应用！请先跑 phase5a_apply.py")
        return False

    if "ARPO_TURN_KL" in src:
        print("⚠️ 5b patch 已 apply，跳过")
        return True

    # 1. 在 ARPO helpers 区域追加 turn-level KL 工具
    helper_code = '''
# ========== Phase 5b ARPO turn-level KL ==========
import torch as _torch_arpo

# turn type id 约定
_TURN_OTHER = 0
_TURN_THINK = 1
_TURN_SEARCH = 2
_TURN_ANSWER = 3
_TURN_INFO = 4

_ARPO_TAG_CACHE = {}   # tokenizer-keyed cache，避免重复编码


def _arpo_build_tag_cache(tokenizer):
    """初始化 tag → token_ids 映射。tokenizer 不变 → cache hit"""
    key = id(tokenizer)
    if key in _ARPO_TAG_CACHE:
        return _ARPO_TAG_CACHE[key]
    cache = {
        'open': {},   # state when entering
        'close': {},
    }
    pairs = [
        ('<think>',       _TURN_THINK,  'open'),
        ('</think>',      _TURN_THINK,  'close'),
        ('<search>',      _TURN_SEARCH, 'open'),
        ('</search>',     _TURN_SEARCH, 'close'),
        ('<answer>',      _TURN_ANSWER, 'open'),
        ('</answer>',     _TURN_ANSWER, 'close'),
        ('<information>', _TURN_INFO,   'open'),
        ('</information>', _TURN_INFO,   'close'),
    ]
    for tag_str, state, mode in pairs:
        ids = tokenizer.encode(tag_str, add_special_tokens=False)
        cache[mode][tuple(ids)] = state
    _ARPO_TAG_CACHE[key] = cache
    return cache


def _arpo_compute_turn_mask(response_ids, tokenizer):
    """扫 response token 序列，标 turn type per-token。返回 (bs, T) long tensor。"""
    cache = _arpo_build_tag_cache(tokenizer)
    open_tags = list(cache['open'].items())   # [((id_seq,), state), ...]
    close_tags = list(cache['close'].items())
    bs, T = response_ids.shape
    out = _torch_arpo.zeros((bs, T), dtype=_torch_arpo.long, device=response_ids.device)
    ids_list = response_ids.tolist()
    for b in range(bs):
        ids = ids_list[b]
        state = _TURN_OTHER
        i = 0
        while i < T:
            matched = False
            # 优先尝试 close tag（结束当前段）
            for tag_seq, _ in close_tags:
                L = len(tag_seq)
                if i + L <= T and tuple(ids[i:i+L]) == tag_seq:
                    # close tag 本身仍在当前 state
                    for j in range(L):
                        out[b, i+j] = state
                    i += L
                    state = _TURN_OTHER
                    matched = True
                    break
            if matched:
                continue
            # 试 open tag
            for tag_seq, new_state in open_tags:
                L = len(tag_seq)
                if i + L <= T and tuple(ids[i:i+L]) == tag_seq:
                    state = new_state
                    for j in range(L):
                        out[b, i+j] = state
                    i += L
                    matched = True
                    break
            if matched:
                continue
            # 普通 token
            out[b, i] = state
            i += 1
    return out


def _arpo_get_beta_per_token(turn_mask):
    """根据 turn type 查 β，返回 (bs, T) float tensor。"""
    import os as _os
    beta_table = {
        _TURN_OTHER:  float(_os.environ.get("ARPO_BETA_OTHER",  "0.001")),
        _TURN_THINK:  float(_os.environ.get("ARPO_BETA_THINK",  "0.001")),
        _TURN_SEARCH: float(_os.environ.get("ARPO_BETA_SEARCH", "0.005")),
        _TURN_ANSWER: float(_os.environ.get("ARPO_BETA_ANSWER", "0.001")),
        _TURN_INFO:   float(_os.environ.get("ARPO_BETA_INFO",   "0.0")),
    }
    out = _torch_arpo.zeros_like(turn_mask, dtype=_torch_arpo.float32)
    for state, beta in beta_table.items():
        out = _torch_arpo.where(turn_mask == state, _torch_arpo.full_like(out, beta), out)
    return out
# ========== Phase 5b ARPO turn-level KL end ==========

'''

    # 在 5a helper end 后插入 5b helper
    marker = "# ========== Phase 5 ARPO helpers end =========="
    if marker not in src:
        print("ERROR: 5a helper end marker not found")
        return False
    idx = src.index(marker) + len(marker)
    idx = src.index("\n", idx) + 1
    src = src[:idx] + helper_code + src[idx:]
    print("✅ 5b helpers inserted")

    # 2. 在 update_policy 的 KL loss 计算处加 turn-level 分支
    # 现有代码:
    #     kld = core_algos.kl_penalty(...)
    #     kl_loss = masked_mean(kld, response_mask)
    # 替换为:
    #     kld = ...
    #     if os.environ.get("ARPO_TURN_KL") == "1":
    #         turn_mask = _arpo_compute_turn_mask(data["responses"], self.tokenizer)
    #         beta_per_tok = _arpo_get_beta_per_token(turn_mask)
    #         kl_loss = (kld * beta_per_tok * response_mask).sum() / response_mask.sum().clamp_min(1.0)
    #         metrics["actor/arpo_turn_kl_used"] = 1
    #     else:
    #         kl_loss = masked_mean(kld, response_mask)

    old = """                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)"""

    new = """                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    # ARPO 5b: turn-level KL (override masked_mean if enabled)
                    if _os_arpo.environ.get("ARPO_TURN_KL") == "1":
                        _turn_mask = _arpo_compute_turn_mask(data["responses"], self.tokenizer)
                        _beta_per_tok = _arpo_get_beta_per_token(_turn_mask)
                        kl_loss = (kld * _beta_per_tok * response_mask).sum() / response_mask.sum().clamp_min(1.0)
                        # NOTE: kl_loss 这里已经把 β 内嵌；下面 policy_loss + kl_loss * kl_loss_coef 会重复乘
                        # 因此需要在 turn_kl 模式下覆盖 self.config.kl_loss_coef = 1.0 effective
                        # 简单做法：返回 kl_loss / self.config.kl_loss_coef 让外部乘回到正确值
                        kl_loss = kl_loss / max(self.config.kl_loss_coef, 1e-6)
                    else:
                        kl_loss = masked_mean(kld, response_mask)"""

    if old not in src:
        print("ERROR: KL loss block not found in dp_actor.py")
        return False
    src = src.replace(old, new)
    print("✅ KL loss turn-aware override inserted")

    open(DP_ACTOR, "w").write(src)
    print("\n=== verify ===")
    src2 = open(DP_ACTOR).read()
    print(f"  ARPO_TURN_KL mentions: {src2.count('ARPO_TURN_KL')}")
    print(f"  _arpo_compute_turn_mask mentions: {src2.count('_arpo_compute_turn_mask')}")
    print(f"  _ARPO_TAG_CACHE mentions: {src2.count('_ARPO_TAG_CACHE')}")
    return True


if __name__ == "__main__":
    ok = patch_dp_actor()
    sys.exit(0 if ok else 1)
