# Phase 3 ToRL — multi-turn tool reward shaping
#
# 复用 qa_em_format 的 5 档 DFA 评分；叠加 per-turn shaping：
#   1. 重复 query 惩罚：相同 search query 出现多次 → 每重复一次 -0.05
#   2. 每 turn 召回奖励：information 块包含 GT → 每命中 +0.05
#   3. clamp ±0.20 防止 shaping 主导原 reward
#
# 部署位置（云端）：
#   verl/utils/reward_score/qa_em_torl.py
#
# 入口路由：verl/trainer/main_ppo_torl.py 把 'nq' → qa_em_torl.compute_score_em

import re
import string
import random


# ---- 复用 qa_em_format 里的工具函数 ----

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence(text):
    # ToRL 增加 max_turns=4 后允许更多 think→search→information→think 循环
    # 状态机不变，但允许循环更深
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)

    if not assistant_match:
        return False, "Missing assistant marker"

    start_pos = assistant_match.end()
    content = text[start_pos:]

    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)

    state = "start"

    for i, part in enumerate(parts):
        if not part.strip():
            continue

        if re.match(r"</?(?:think|search|information|answer)>", part):
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                pass
            elif state in ["start", "after_think", "after_search", "information"]:
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"

    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def extract_solution(solution_str):
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if len(matches) <= 1:
        return None
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def extract_search_queries(text: str) -> list:
    """ToRL 新增：抽取 <search>query</search> 里的 query 文本"""
    pattern = r"<search>(.*?)</search>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list) -> bool:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


# ---- ToRL 新增：per-turn shaping ----

def compute_torl_shaping(solution_str, ground_truth,
                         repeat_query_penalty=-0.05,
                         per_turn_hit_bonus=0.05,
                         shaping_clamp=0.20):
    """
    返回 (shaping_score, debug_info)
      - 每个重复 query: -0.05
      - 每个 information 块命中 GT: +0.05
      - clamp 到 [-clamp, +clamp]
    """
    queries = extract_search_queries(solution_str)
    info_blocks = extract_information_blocks(solution_str)
    targets = ground_truth.get('target', [])
    if isinstance(targets, str):
        targets = [targets]
    targets_norm = [normalize_answer(t) for t in targets]

    # (1) 重复 query 惩罚
    seen = set()
    repeat_pen = 0
    repeat_count = 0
    for q in queries:
        q_norm = normalize_answer(q)
        if q_norm in seen:
            repeat_pen += repeat_query_penalty
            repeat_count += 1
        else:
            seen.add(q_norm)

    # (2) 每 turn 召回奖励
    per_turn_bonus = 0
    hit_count = 0
    for info in info_blocks:
        info_norm = normalize_answer(info)
        if any(t in info_norm for t in targets_norm):
            per_turn_bonus += per_turn_hit_bonus
            hit_count += 1

    raw_shaping = repeat_pen + per_turn_bonus
    clamped = max(-shaping_clamp, min(shaping_clamp, raw_shaping))

    return clamped, {
        'n_queries': len(queries),
        'n_unique_queries': len(seen),
        'n_repeat_queries': repeat_count,
        'n_info_blocks': len(info_blocks),
        'n_per_turn_hits': hit_count,
        'raw_shaping': raw_shaping,
        'clamped_shaping': clamped,
    }


# ---- 主入口：与 qa_em_format.compute_score_em 同签名 + ToRL shaping ----

def compute_score_em(solution_str, ground_truth, method='strict',
                     structure_format_score=0, final_format_score=0,
                     retrieval_score=0, format_score=0, score=1.,
                     # ToRL shaping params (configurable via reward_model.* in CLI)
                     repeat_query_penalty=-0.05,
                     per_turn_hit_bonus=0.05,
                     shaping_clamp=0.20):
    """
    Phase 3 ToRL composite reward:
      base = qa_em_format 5档分级
      + per-turn shaping (clamp ±0.20)
    """
    # ---- base reward (与 qa_em_format 完全相同) ----
    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str[:500]}...")  # 截断防 log 爆炸

    if answer is None:
        if is_valid_format:
            base = structure_format_score + retrieval_score if retrieval_correct else structure_format_score
        else:
            base = 0
    else:
        if em_check(answer, ground_truth['target']):
            base = score if is_valid_format else (score - structure_format_score)
        elif is_valid_format:
            base = structure_format_score + retrieval_score if retrieval_correct else structure_format_score
        else:
            base = final_format_score

    # ---- ToRL shaping ----
    shaping, dbg = compute_torl_shaping(
        solution_str, ground_truth,
        repeat_query_penalty=repeat_query_penalty,
        per_turn_hit_bonus=per_turn_hit_bonus,
        shaping_clamp=shaping_clamp,
    )

    final_reward = base + shaping

    if do_print:
        print(f"[ToRL shaping] base={base:.3f} shaping={shaping:+.3f} → final={final_reward:.3f}")
        print(f"               n_queries={dbg['n_queries']} unique={dbg['n_unique_queries']} repeat={dbg['n_repeat_queries']} info_hits={dbg['n_per_turn_hits']}")

    return final_reward
