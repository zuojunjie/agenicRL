"""Reward function for MATH+Python env.

extract_solution: pulls last <answer>...</answer> from chat history (skip prompt example)
compute_score:    binary EM, with sympy fallback for math equivalence
                  (e.g. "1/2" == "0.5", "\\frac{1}{2}" == "0.5")

For Phase 10 baseline: pure binary (no format reward, since 7.8e proved format reward
hurts on NQ; we'll re-evaluate on MATH when adding shaping in Phase 11).

Path target: skyrl-gym/skyrl_gym/envs/math_python/utils.py
"""
from __future__ import annotations

import re
import string
from typing import List, Optional, Union


def normalize_answer(s: str) -> str:
    """Normalize: lowercase, strip articles, strip punct, collapse spaces."""
    s = s.lower()
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # collapse whitespace
    s = " ".join(s.split())
    return s


def extract_solution(solution_str: str) -> Optional[str]:
    """Extract the LAST <answer>...</answer> from text.

    Falls back to \\boxed{...} if no <answer> tag found (paper convention).

    Returns None if neither pattern matches.
    """
    # Primary: <answer>...</answer>
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()

    # Fallback: \boxed{...}
    # Match balanced braces (one level deep is fine for most math answers)
    boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = list(re.finditer(boxed_pattern, solution_str))
    if matches:
        return matches[-1].group(1).strip()

    return None


def _sympy_equivalent(a: str, b: str) -> bool:
    """Try sympy-based math equivalence. Returns False on parse failure."""
    try:
        import sympy as sp
        # Try parsing both as math expressions
        try:
            ea = sp.sympify(a)
            eb = sp.sympify(b)
        except Exception:
            # Try latex parsing
            try:
                from sympy.parsing.latex import parse_latex
                ea = parse_latex(a) if "\\" in a else sp.sympify(a)
                eb = parse_latex(b) if "\\" in b else sp.sympify(b)
            except Exception:
                return False
        diff = sp.simplify(ea - eb)
        return diff == 0
    except Exception:
        return False


def em_check(prediction: str, golden_answers: Union[str, List[str]]) -> int:
    """1 if prediction matches any gold answer (normalized + sympy fallback)."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    if hasattr(golden_answers, "tolist"):
        golden_answers = golden_answers.tolist()

    p_norm = normalize_answer(prediction)
    for gold in golden_answers:
        g_norm = normalize_answer(gold)
        if g_norm == p_norm:
            return 1
        # sympy fallback for math expressions
        if _sympy_equivalent(prediction.strip(), str(gold).strip()):
            return 1
    return 0


def compute_score(
    solution_str: str,
    ground_truth,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """Binary EM with sympy fallback. Used in train + eval (Phase 10 baseline).

    Args:
        solution_str: full chat history concatenation
        ground_truth: dict with key 'target' (list/str) or just str/list
        method:       unused (compat with SkyRL signature)
        format_score: returned when answer extracted but wrong (default 0)
        score:        returned when correct (default 1)
    """
    answer = extract_solution(solution_str)
    if answer is None:
        return 0.0

    # Normalize ground_truth shape
    if isinstance(ground_truth, dict):
        target = ground_truth.get("target", ground_truth.get("answer"))
    else:
        target = ground_truth

    if em_check(answer, target):
        return score
    return format_score


# Self-test
if __name__ == "__main__":
    tests = [
        ("<answer>42</answer>", "42", 1.0),
        ("<answer>42</answer>", "43", 0.0),
        ("\\boxed{42}", "42", 1.0),
        # sympy equivalence
        ("<answer>1/2</answer>", "0.5", 1.0),
        ("<answer>2*x + 4</answer>", "2*(x+2)", 1.0),
        # no answer
        ("just thinking, no answer", "42", 0.0),
        # multiple answer tags - takes last
        ("<answer>X</answer> some thinking <answer>42</answer>", "42", 1.0),
    ]
    for sol, gt, expected in tests:
        got = compute_score(sol, {"target": [gt]})
        status = "✅" if abs(got - expected) < 0.01 else "❌"
        print(f"{status} compute_score({sol!r}, gt={gt!r}) = {got}, expected {expected}")
