#!/usr/bin/env python3
"""Analyze SkyRL eval dump to figure out which DFA branch each sample falls into."""
import json
import sys

sys.path.insert(0, "/root/autodl-tmp/external/SkyRL/skyrl-gym")
from skyrl_gym.envs.search.utils import (
    _dfa_is_valid_sequence,
    _dfa_is_retrieval_correct,
    em_check,
    extract_solution,
)

F = sys.argv[1]

n_total = n_em = n_format = n_format_and_em = n_retrieval_hit = 0
n_no_answer = 0
n_answer_extracted = 0

# Reward distribution per DFA branch
reward_buckets = {0.0: 0, 0.1: 0, 0.2: 0, 0.3: 0, 0.8: 0, 1.0: 0}

with open(F) as f:
    for line in f:
        d = json.loads(line)
        n_total += 1
        sol = d["input_prompt"] + d["output_response"]
        gt = d["env_extras"]["reward_spec"]["ground_truth"]["target"]
        ans = extract_solution(sol)
        is_valid, _ = _dfa_is_valid_sequence(sol)

        if ans is None:
            n_no_answer += 1
        else:
            n_answer_extracted += 1

        if ans is not None and em_check(ans, gt):
            n_em += 1
        if is_valid:
            n_format += 1
            if _dfa_is_retrieval_correct(sol, gt):
                n_retrieval_hit += 1
            if ans is not None and em_check(ans, gt):
                n_format_and_em += 1

        # Replicate compute_score_dfa logic
        retrieval_ok = is_valid and _dfa_is_retrieval_correct(sol, gt)
        if ans is None:
            if is_valid:
                if retrieval_ok:
                    r = 0.3
                else:
                    r = 0.2
            else:
                r = 0.0
        else:
            if em_check(ans, gt):
                if is_valid:
                    r = 1.0
                else:
                    r = 0.8
            elif is_valid:
                if retrieval_ok:
                    r = 0.3
                else:
                    r = 0.2
            else:
                r = 0.1
        reward_buckets[r] = reward_buckets.get(r, 0) + 1

print(f"Total: {n_total}")
print(f"  no answer extracted:   {n_no_answer:5d}  ({n_no_answer/n_total*100:.1f}%)")
print(f"  answer extracted:      {n_answer_extracted:5d}  ({n_answer_extracted/n_total*100:.1f}%)")
print(f"  EM correct (any fmt):  {n_em:5d}  ({n_em/n_total*100:.1f}%)")
print(f"  is_valid_format=True:  {n_format:5d}  ({n_format/n_total*100:.1f}%)")
print(f"    → AND EM correct:    {n_format_and_em:5d}  ({n_format_and_em/n_total*100:.1f}%)")
print(f"    → AND retrieval hit: {n_retrieval_hit:5d}  ({n_retrieval_hit/n_total*100:.1f}%)")
print()
print("Reward bucket distribution (predicted by DFA logic):")
for r in sorted(reward_buckets):
    pct = reward_buckets[r] / n_total * 100
    print(f"  {r:.1f}:  {reward_buckets[r]:5d}  ({pct:5.1f}%)")
total_reward = sum(r * cnt for r, cnt in reward_buckets.items())
print(f"\nPredicted avg reward: {total_reward/n_total:.4f}")
