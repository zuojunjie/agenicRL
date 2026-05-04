"""mmap-based drop-in replacement for the in-memory corpus list.

Patches retrieval_server.py: replace load_corpus() body with MmapCorpus.
MmapCorpus[idx] returns the same dict that the original list would, but
the underlying jsonl bytes live in page cache (kernel-evictable, NOT counted
against cgroup memory.max), saving ~25-28 GB of anonymous heap on the
21M-passage Wikipedia corpus.
"""
import json
import mmap
import os

import numpy as np


class MmapCorpus:
    """List-like view over a jsonl file backed by mmap + offset index.

    Drop-in replacement for `corpus = [json.loads(line) for line in jsonl]`.
    Supports `corpus[idx]` (returns parsed dict) and `len(corpus)`.
    """

    def __init__(self, jsonl_path: str, offsets_path: str = None):
        self.jsonl_path = jsonl_path
        if offsets_path is None:
            offsets_path = jsonl_path + ".offsets.npy"
        if not os.path.exists(offsets_path):
            raise FileNotFoundError(
                f"offsets index missing: {offsets_path}\n"
                f"build it once with: python build_corpus_offsets.py {jsonl_path}"
            )

        self.offsets = np.load(offsets_path, mmap_mode="r")  # also mmap'd
        self._n = len(self.offsets) - 1

        self._fd = open(jsonl_path, "rb")
        self._mm = mmap.mmap(self._fd.fileno(), 0, prot=mmap.PROT_READ)

        print(f"[MmapCorpus] {self._n} passages from {jsonl_path}", flush=True)
        print(f"[MmapCorpus] offsets: {offsets_path} ({os.path.getsize(offsets_path)/1e6:.0f} MB)", flush=True)
        print(f"[MmapCorpus] anon heap: ~0 GB (vs ~30 GB for in-memory list)", flush=True)

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Used rarely (we don't expect slicing in retriever code, but be safe)
            return [self[i] for i in range(*idx.indices(self._n))]
        i = int(idx)
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError(i)
        start = int(self.offsets[i])
        end = int(self.offsets[i + 1])
        # strip trailing \n if present
        line = self._mm[start:end]
        if line.endswith(b"\n"):
            line = line[:-1]
        return json.loads(line)

    def __iter__(self):
        for i in range(self._n):
            yield self[i]


# Convenience: a load_corpus() drop-in that returns MmapCorpus
def load_corpus_mmap(corpus_path: str):
    return MmapCorpus(corpus_path)
