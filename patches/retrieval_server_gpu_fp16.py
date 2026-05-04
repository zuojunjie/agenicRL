"""Patch retrieval_server.py to use GpuIndexFlatConfig fp16 direct path.

Reason: FAISS cloner (`index_cpu_to_all_gpus + co.useFloat16=True`) silently
ignores fp16 for IndexFlat, leaving the GPU index at fp32 (64GB on a 96GB card,
no room for the trainer). The direct GpuIndexFlatConfig path with useFloat16=True
honors fp16 (32GB), tested at 99.4% recall vs fp32 (see memory note
`faiss_fp16_flat_for_7b.md`).

Apply on cloud after reboot:
    python3 /root/autodl-tmp/agenicRL/patches/retrieval_server_gpu_fp16.py
Then restart retrieval_server with `--faiss_gpu` (single GPU is fine).
"""

path = "/root/autodl-tmp/external/Search-R1/search_r1/search/retrieval_server.py"
src = open(path).read()

old = """        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)"""

new = """        if config.faiss_gpu:
            # AUTOPATCH: GpuIndexFlatConfig direct fp16 path (cloner silently ignores fp16 for IndexFlat).
            # Verified 99.4% recall vs fp32 — see notes/faiss_fp16_flat_for_7b.md
            cpu_idx = self.index
            is_flat = isinstance(
                cpu_idx,
                (faiss.IndexFlat, faiss.IndexFlatIP, faiss.IndexFlatL2),
            )
            if is_flat:
                res = faiss.StandardGpuResources()
                cfg = faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = True
                cfg.device = 0
                if cpu_idx.metric_type == faiss.METRIC_L2:
                    gpu_idx = faiss.GpuIndexFlatL2(res, cpu_idx.d, cfg)
                else:
                    gpu_idx = faiss.GpuIndexFlatIP(res, cpu_idx.d, cfg)
                # transfer all vectors at once (Flat allows reconstruct_n)
                vecs = cpu_idx.reconstruct_n(0, cpu_idx.ntotal)
                gpu_idx.add(vecs)
                del vecs, cpu_idx
                self.index = gpu_idx
                self._faiss_resources = res  # keep alive
                print(
                    f"[retrieval_server] GpuIndexFlat fp16 active "
                    f"(d={gpu_idx.d}, ntotal={gpu_idx.ntotal}, ~32GB VRAM)"
                )
            else:
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.shard = True
                self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)"""

if new in src:
    print("already patched")
elif old not in src:
    print("PATTERN NOT FOUND — manual inspection needed")
else:
    open(path, "w").write(src.replace(old, new))
    print("PATCHED retrieval_server.py: GpuIndexFlatConfig fp16 direct path")
    print("Restart server with `--faiss_gpu` (single GPU works).")
