#!/usr/bin/env python3
"""Patch SkyRL search utils.py:
1. _dfa_is_valid_sequence: handle both verl (with assistant marker) and SkyRL (without).
   Run full state-machine validation in 4090-strict mode if marker present, else lenient
   structural check (think + search + answer all balanced and present).
2. extract_solution: require >=2 <answer> matches (skip the prompt's example "<answer> Beijing </answer>").
"""
import re
import sys

f = sys.argv[1]
src = open(f).read()

# Replace _dfa_is_valid_sequence with a version that works in both setups
new_is_valid = '''def _dfa_is_valid_sequence(text):
    """patch (agenicRL): work in both verl (with <|im_start|>assistant marker)
    and SkyRL search env (chat_history contents joined, no markers)."""
    import re
    assistant_match = re.search(r"<\\\\|im_start\\\\|>assistant\\\\s*", text)
    if assistant_match:
        # 4090 verl strict mode
        start_pos = assistant_match.end()
        content = text[start_pos:]
        end_match = re.search(r"<\\\\|im_end\\\\|>", content)
        if end_match:
            content = content[:end_match.start()]
    else:
        # SkyRL lenient mode: skip the prompt example by taking text AFTER the
        # first </answer> (which is the example) if there is more content after.
        first_close = text.find("</answer>")
        if first_close >= 0:
            tail = text[first_close + len("</answer>"):]
            if "<answer>" in tail and "</answer>" in tail:
                content = tail  # skip the example
            else:
                content = text  # only one answer total → the example, no real answer
        else:
            content = text

    # Check balanced tags
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} open vs {closing_count} close"

    # Must have at least one <think> and one <answer>
    has_answer = "<answer>" in content and "</answer>" in content
    has_think = "<think>" in content and "</think>" in content
    if not (has_answer and has_think):
        return False, "missing <think> or <answer>"

    return True, "valid"
'''

src = re.sub(
    r"def _dfa_is_valid_sequence\(text\):.*?return has_answer, \(\"ok\" if has_answer else \"no <answer>\"\)\n",
    new_is_valid,
    src,
    count=1,
    flags=re.DOTALL,
)

# Patch extract_solution to require >=2 matches (skip prompt's example)
new_extract = '''def extract_solution(solution_str):
    """patch (agenicRL): require >=2 <answer> matches; skip the prompt example.
    The prompt template contains "For example, <answer> Beijing </answer>" — that's
    the FIRST match. Real model answer is the LAST match if there's a 2nd."""
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if len(matches) < 2:
        return None
    return matches[-1].group(1).strip()
'''

src = re.sub(
    r'def extract_solution\(solution_str\):\n    """Extract the equation from the solution string\."""\n.*?return matches\[-1\]\.group\(1\)\.strip\(\)\n',
    new_extract,
    src,
    count=1,
    flags=re.DOTALL,
)

open(f, "w").write(src)
print("patched OK")
