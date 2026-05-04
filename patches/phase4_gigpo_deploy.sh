#!/usr/bin/env bash
# Phase 4 GiGPO 部署：apply patch + 修改 call site (ray_trainer.py)
#
# 用法（云端）：
#   bash /root/autodl-tmp/agenicRL/patches/phase4_gigpo_deploy.sh
#
# 回滚：
#   bash /root/autodl-tmp/agenicRL/patches/phase4_gigpo_deploy.sh revert

set -e

VERL=/root/autodl-tmp/external/Search-R1
PATCH=/root/autodl-tmp/agenicRL/patches/phase4_gigpo_two_layer.patch

if [ "${1:-}" = "revert" ]; then
    echo "===  Reverting GiGPO ==="
    cd $VERL
    patch -p1 -R < $PATCH 2>&1 | head -3
    # 撤销 ray_trainer.py 的 gigpo_alpha 参数
    python << "PYEOF"
path = "/root/autodl-tmp/external/Search-R1/verl/trainer/ppo/ray_trainer.py"
src = open(path).read()
old = """advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        gigpo_alpha=getattr(__import__('os').environ.get('GIGPO_ALPHA') and float(__import__('os').environ['GIGPO_ALPHA']), '__call__', None) if False else None)"""
new = """advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)"""
# 简单恢复
src = src.replace("                                                                        gigpo_alpha=__import__('os').environ.get('GIGPO_ALPHA') and float(__import__('os').environ.get('GIGPO_ALPHA', 'nan')) or None,\n", "")
open(path, "w").write(src)
PYEOF
    echo "✅ reverted"
    exit 0
fi

echo "=== 1. patch core_algos.py ==="
cd $VERL
patch -p1 --dry-run < $PATCH 2>&1 | head -3
patch -p1 < $PATCH 2>&1 | head -3
grep -c "gigpo_alpha" verl/trainer/ppo/core_algos.py

echo
echo "=== 2. modify ray_trainer.py call site ==="
# 把 compute_grpo_outcome_advantage(...) 调用加 gigpo_alpha 参数
# 用环境变量传，最简单（脚本里 export GIGPO_ALPHA=0.7）
python << "PYEOF"
path = "/root/autodl-tmp/external/Search-R1/verl/trainer/ppo/ray_trainer.py"
src = open(path).read()
old = """        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)"""
new = """        import os as _os
        _gigpo_alpha = float(_os.environ['GIGPO_ALPHA']) if _os.environ.get('GIGPO_ALPHA') else None
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        gigpo_alpha=_gigpo_alpha)"""
assert old in src, "old call site not found"
src = src.replace(old, new)
open(path, "w").write(src)
print("✅ ray_trainer.py 已加 GIGPO_ALPHA env-var hook")
PYEOF

echo
echo "=== 3. 验证 ==="
grep -c "gigpo_alpha\|GIGPO_ALPHA" $VERL/verl/trainer/ppo/core_algos.py $VERL/verl/trainer/ppo/ray_trainer.py
echo
echo "✅ 部署完成。在训练脚本前 \`export GIGPO_ALPHA=0.7\` 即可启用 GiGPO；不设则退化 GRPO。"
