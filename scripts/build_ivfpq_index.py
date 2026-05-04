#!/usr/bin/env python
"""Rebuild Wikipedia FAISS index from Flat (64GB fp32) → IVF_PQ (~1.3GB).

Source:  /root/autodl-tmp/data/wikipedia_index/e5_Flat.index  (21M × 768 fp32)
Target:  /root/autodl-tmp/data/wikipedia_index/e5_IVFPQ.index (~1.3GB)

Index params:
  nlist=4096   (~sqrt(21M))
  m=64         (768 dim / 12 = 64 PQ subquantizers)
  nbits=8      (1 byte per code)
  nprobe=32    (default search)
  metric=IP    (preserves e5's inner-product semantics)

Memory peak: ~5GB (1.5GB temp chunk + 1.3GB index + 1M training sample 3GB).
Designed to coexist with sshd (chunked + can be ionice/nice'd at launch).

Run with:
  ionice -c 3 nice -n 19 python build_ivfpq_index.py 2>&1 | tee /tmp/ivfpq_build.log
"""

import faiss
import numpy as np
import time
import os
import gc

FLAT_PATH = "/root/autodl-tmp/data/wikipedia_index/e5_Flat.index"
OUT_PATH = "/root/autodl-tmp/data/wikipedia_index/e5_IVFPQ.index"

NLIST = 4096
M = 64
NBITS = 8
NPROBE = 32
TRAIN_SAMPLE = 1_000_000
CHUNK = 500_000


def main():
    if os.path.exists(OUT_PATH):
        sz = os.path.getsize(OUT_PATH) / 1e9
        print(f"[skip] {OUT_PATH} already exists ({sz:.2f}GB). Delete to rebuild.")
        return

    print(f"=== load source Flat (mmap) ===")
    t0 = time.time()
    src = faiss.read_index(FLAT_PATH, faiss.IO_FLAG_MMAP)
    d = src.d
    ntotal = src.ntotal
    metric_str = "IP" if src.metric_type == faiss.METRIC_INNER_PRODUCT else "L2"
    print(f"  d={d}, ntotal={ntotal:,}, metric={metric_str} ({time.time()-t0:.1f}s)")

    # Use first 1M vectors as training sample (sequential read, fast)
    print(f"\n=== sample first {TRAIN_SAMPLE:,} vectors for IVF_PQ training ===")
    t1 = time.time()
    train_vecs = src.reconstruct_n(0, TRAIN_SAMPLE).astype(np.float32, copy=False)
    print(f"  sampled in {time.time()-t1:.1f}s, shape={train_vecs.shape}")

    print(f"\n=== build empty IVF_PQ (nlist={NLIST}, m={M}, nbits={NBITS}) ===")
    if src.metric_type == faiss.METRIC_INNER_PRODUCT:
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, NLIST, M, NBITS)
        index.metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, NLIST, M, NBITS)

    print(f"=== train ===")
    t2 = time.time()
    index.train(train_vecs)
    print(f"  trained in {time.time()-t2:.1f}s")
    del train_vecs
    gc.collect()

    print(f"\n=== add all {ntotal:,} vectors in chunks of {CHUNK:,} ===")
    t3 = time.time()
    last_log = t3
    for start in range(0, ntotal, CHUNK):
        n = min(CHUNK, ntotal - start)
        chunk = src.reconstruct_n(start, n)
        index.add(chunk)
        del chunk
        now = time.time()
        if now - last_log >= 20.0 or (start + n) >= ntotal:
            pct = (start + n) / ntotal * 100
            elapsed = now - t3
            rate = (start + n) / max(elapsed, 1)
            eta = (ntotal - start - n) / max(rate, 1)
            print(
                f"  {start+n:,}/{ntotal:,} ({pct:.1f}%) "
                f"@ {elapsed:.0f}s, rate={rate/1e3:.1f}K/s, ETA={eta/60:.1f}min",
                flush=True,
            )
            last_log = now

    index.nprobe = NPROBE
    print(f"\n=== save to {OUT_PATH} ===")
    t4 = time.time()
    faiss.write_index(index, OUT_PATH)
    sz = os.path.getsize(OUT_PATH) / 1e9
    print(f"  saved {sz:.2f}GB in {time.time()-t4:.1f}s")
    print(f"\n=== TOTAL: {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
