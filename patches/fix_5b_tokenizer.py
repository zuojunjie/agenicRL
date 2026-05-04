"""修 5b patch 的 tokenizer bug：换成 module-level lazy load"""

path = "/root/autodl-tmp/external/Search-R1/verl/workers/actor/dp_actor.py"
src = open(path).read()

old_helper = '''def _arpo_build_tag_cache(tokenizer):
    """初始化 tag → token_ids 映射。tokenizer 不变 → cache hit"""
    key = id(tokenizer)
    if key in _ARPO_TAG_CACHE:
        return _ARPO_TAG_CACHE[key]
    cache = {
        \'open\': {},   # state when entering
        \'close\': {},
    }
    pairs = [
        (\'<think>\',       _TURN_THINK,  \'open\'),
        (\'</think>\',      _TURN_THINK,  \'close\'),
        (\'<search>\',      _TURN_SEARCH, \'open\'),
        (\'</search>\',     _TURN_SEARCH, \'close\'),
        (\'<answer>\',      _TURN_ANSWER, \'open\'),
        (\'</answer>\',     _TURN_ANSWER, \'close\'),
        (\'<information>\', _TURN_INFO,   \'open\'),
        (\'</information>\', _TURN_INFO,   \'close\'),
    ]
    for tag_str, state, mode in pairs:
        ids = tokenizer.encode(tag_str, add_special_tokens=False)
        cache[mode][tuple(ids)] = state
    _ARPO_TAG_CACHE[key] = cache
    return cache'''

new_helper = '''_ARPO_TOKENIZER_PATH = _os_arpo.environ.get("ARPO_TOKENIZER_PATH", "/root/autodl-tmp/models/Qwen2.5-3B-Instruct")

def _arpo_build_tag_cache():
    """Lazy load tokenizer + build tag cache. Module-level singleton（不依赖 actor.self.tokenizer）"""
    if "main" in _ARPO_TAG_CACHE:
        return _ARPO_TAG_CACHE["main"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(_ARPO_TOKENIZER_PATH, trust_remote_code=True)
    cache = {"open": {}, "close": {}}
    pairs = [
        ("<think>",       _TURN_THINK,  "open"),
        ("</think>",      _TURN_THINK,  "close"),
        ("<search>",      _TURN_SEARCH, "open"),
        ("</search>",     _TURN_SEARCH, "close"),
        ("<answer>",      _TURN_ANSWER, "open"),
        ("</answer>",     _TURN_ANSWER, "close"),
        ("<information>", _TURN_INFO,   "open"),
        ("</information>", _TURN_INFO,   "close"),
    ]
    for tag_str, state, mode in pairs:
        ids = tokenizer.encode(tag_str, add_special_tokens=False)
        cache[mode][tuple(ids)] = state
    _ARPO_TAG_CACHE["main"] = cache
    print(f"[ARPO 5b] tag cache built: {sum(len(v) for v in cache.values())} tag patterns")
    return cache'''

assert old_helper in src, "ERROR: old _arpo_build_tag_cache not found"
src = src.replace(old_helper, new_helper)
print("OK replaced _arpo_build_tag_cache")

old_call = '''def _arpo_compute_turn_mask(response_ids, tokenizer):
    """扫 response token 序列，标 turn type per-token。返回 (bs, T) long tensor。"""
    cache = _arpo_build_tag_cache(tokenizer)'''

new_call = '''def _arpo_compute_turn_mask(response_ids):
    """扫 response token 序列，标 turn type per-token。返回 (bs, T) long tensor。"""
    cache = _arpo_build_tag_cache()'''

assert old_call in src, "ERROR: old _arpo_compute_turn_mask not found"
src = src.replace(old_call, new_call)
print("OK replaced _arpo_compute_turn_mask signature")

old_invocation = '_turn_mask = _arpo_compute_turn_mask(data["responses"], self.tokenizer)'
new_invocation = '_turn_mask = _arpo_compute_turn_mask(data["responses"])'

assert old_invocation in src, "ERROR: old invocation not found"
src = src.replace(old_invocation, new_invocation)
print("OK replaced invocation in update_policy")

open(path, "w").write(src)
print("=== verify ===")
src2 = open(path).read()
n_self_tok = src2.count("self.tokenizer")
n_build = src2.count("_arpo_build_tag_cache()")
n_compute = src2.count("_arpo_compute_turn_mask(data")
print("self.tokenizer remaining:", n_self_tok)
print("_arpo_build_tag_cache():", n_build)
print("_arpo_compute_turn_mask(data:", n_compute)
