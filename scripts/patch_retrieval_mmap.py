"""Patch retrieval_server.py to use MmapCorpus (saves ~27GB CPU RAM)."""
import re
import sys

f = sys.argv[1]
src = open(f).read()

import_line = ('import sys; sys.path.insert(0, "/root/autodl-tmp/agenicRL/scripts"); '
               'from mmap_corpus_patch import MmapCorpus  # patch (agenicRL)\n')
if "from mmap_corpus_patch" not in src:
    src = src.replace("import datasets\n", "import datasets\n" + import_line, 1)

new_load = '''def load_corpus(corpus_path: str):
    """patch (agenicRL): mmap-backed, anon heap ~0 vs ~30 GB"""
    return MmapCorpus(corpus_path)
'''

src = re.sub(
    r"def load_corpus\(corpus_path: str\):.*?return corpus\n",
    new_load,
    src,
    count=1,
    flags=re.DOTALL,
)
# Also handle if existing implementation has different signature
src = re.sub(
    r"def load_corpus\(.*?\):\n.*?return corpus\n",
    new_load,
    src,
    count=1,
    flags=re.DOTALL,
)

open(f, "w").write(src)
print("patched OK")
