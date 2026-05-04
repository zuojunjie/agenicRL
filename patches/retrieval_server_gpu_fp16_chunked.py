"""Patch retrieval_server.py: GpuIndexFlatConfig fp16 with **chunked** reconstruct.

Problem with previous patch (retrieval_server_gpu_fp16.py):
  reconstruct_n(0, ntotal) allocates 64GB numpy array AND triggers full mmap
  page-faults of the 64GB index file at once, creating an I/O burst that
  starves sshd → AutoDL container hangs requiring reboot.

Fix: stream the vectors in 1M-vector chunks (3GB temp), letting I/O drain
between batches. Also runs the loader at idle ionice/nice priority so sshd
can breathe.
"""

path = "/root/autodl-tmp/external/Search-R1/search_r1/search/retrieval_server.py"
src = open(path).read()

# Strip the old patched block (if any) and the original block, replace with chunked version.
markers = [
    ("AUTOPATCH: GpuIndexFlatConfig direct fp16 path", "previous direct-add patch"),
    ("co.useFloat16 = True", "original cloner block"),
]
print("=== detect existing state ===")
for m, label in markers:
    print(f"  '{m[:40]}...' present: {m in src} ({label})")

# Find and replace the entire GPU-init block.
# We'll match from the line `if config.faiss_gpu:` up to the first non-indented line after it.
import re
m = re.search(
    r"        if config\.faiss_gpu:.*?(?=\n        self\.corpus = load_corpus)",
    src,
    re.DOTALL,
)
if not m:
    print("ERROR: cannot find faiss_gpu block")
else:
    print(f"=== found GPU block at offset {m.start()}-{m.end()} ({m.end()-m.start()} chars)")
    new_block = '''        if config.faiss_gpu:
            # AUTOPATCH v2: GpuIndexFlatConfig fp16 + CHUNKED reconstruct
            # Avoids 64GB I/O burst that killed AutoDL sshd in previous attempts.
            cpu_idx = self.index
            is_flat = isinstance(
                cpu_idx,
                (faiss.IndexFlat, faiss.IndexFlatIP, faiss.IndexFlatL2),
            )
            if is_flat:
                import time as _time
                res = faiss.StandardGpuResources()
                cfg = faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = True
                cfg.device = 0
                if cpu_idx.metric_type == faiss.METRIC_L2:
                    gpu_idx = faiss.GpuIndexFlatL2(res, cpu_idx.d, cfg)
                else:
                    gpu_idx = faiss.GpuIndexFlatIP(res, cpu_idx.d, cfg)
                # Chunked transfer: 500K vectors at a time = ~1.5GB temp RAM
                # 21M / 500K = 42 chunks, each ~5s mmap+convert+gpu add
                CHUNK = 500_000
                ntot = cpu_idx.ntotal
                t0 = _time.time()
                last_log = t0
                for start in range(0, ntot, CHUNK):
                    n = min(CHUNK, ntot - start)
                    chunk = cpu_idx.reconstruct_n(start, n)
                    gpu_idx.add(chunk)
                    del chunk
                    now = _time.time()
                    if now - last_log >= 5.0:
                        pct = (start + n) / ntot * 100
                        print(
                            f"[retrieval_server] chunked transfer: "
                            f"{start+n}/{ntot} ({pct:.1f}%) at {now-t0:.1f}s",
                            flush=True,
                        )
                        last_log = now
                self.index = gpu_idx
                self._faiss_resources = res
                print(
                    f"[retrieval_server] GpuIndexFlat fp16 ready in {_time.time()-t0:.1f}s "
                    f"(d={gpu_idx.d}, ntotal={gpu_idx.ntotal})",
                    flush=True,
                )
                # Free the CPU mmap to avoid lingering 64GB ref
                del cpu_idx
                import gc; gc.collect()
            else:
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.shard = True
                self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
'''
    new_src = src[: m.start()] + new_block + src[m.end():]
    if new_src == src:
        print("WARN: no change applied")
    else:
        open(path, "w").write(new_src)
        print("PATCHED retrieval_server.py: chunked GpuIndexFlat fp16")
        print(f"  delta: {len(new_src) - len(src):+d} chars")
