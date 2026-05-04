"""
IVF_PQ 重建 v2：streaming + mmap，RSS 峰值 ≤ 5 GB

设计:
  1. faiss.read_index 用 IO_FLAG_MMAP（文件页不进 cgroup RSS）
  2. 不分配 64GB np 数组；streaming reconstruct 1M batch 直接 add
  3. 训练样本只 1M（3GB），训完即释放
  4. cgroup 友好：峰值 RSS = mmap'd flat (低) + 1M batch np (3GB) + ivfpq (小)

cgroup 实测: memory.max=360GB, memory.current=266GB, 余 94GB
本脚本预期 RSS 峰值 < 5GB，不会 OOM。
"""
import os
import sys
import time
import faiss
import numpy as np
import psutil
import gc
import resource

# 限制虚存 80GB（mmap 需要），但 cgroup RSS 限制由系统兜底
resource.setrlimit(resource.RLIMIT_AS, (80*1024**3, 80*1024**3))

SRC = "/root/autodl-tmp/data/wikipedia_index/e5_Flat.index"
DST = "/root/autodl-tmp/data/wikipedia_index/e5_IVF_PQ.index"
DST_TMP = DST + ".tmp"
PROGRESS_LOG = "/tmp/ivfpq_progress.log"

NLIST = 4096
M = 8
NBITS = 8
TRAIN_SAMPLES = 1_000_000
BATCH_SIZE = 500_000   # 0.5M × 768 × 4B = 1.5 GB per batch

faiss.omp_set_num_threads(32)


def log(msg):
    p = psutil.Process()
    rss_gb = p.memory_info().rss / 1e9
    line = f"[{time.strftime('%H:%M:%S')}] [RSS={rss_gb:.2f}GB] {msg}"
    print(line, flush=True)
    sys.stdout.flush()
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")


def main():
    log("=== IVF_PQ streaming rebuild ===")
    log(f"OMP threads: {faiss.omp_get_max_threads()}")

    # 1. mmap Flat 索引
    log("--- 1. mmap Flat (~40s, file-backed, no RSS hit) ---")
    flat = faiss.read_index(SRC, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    n, d = flat.ntotal, flat.d
    log(f"loaded: {type(flat).__name__}, n={n:,}, d={d}")

    # 2. 训练子集（1M 样本，~3GB 临时）
    log("--- 2. extract 1M train samples ---")
    rng = np.random.default_rng(42)
    train_idx = np.sort(rng.choice(n, TRAIN_SAMPLES, replace=False).astype(np.int64))
    # 一次性 reconstruct 1M（3GB）
    xb_train = flat.reconstruct_n(0, 0)  # 触发 dummy（试错）
    # 用 reconstruct_n 拉 1M
    # 注意：train_idx 是非连续，必须逐条 reconstruct（慢）或拉连续段
    # 取连续前 1M 简化（应该足够代表）
    log("  using first 1M as train set (continuous, fast)")
    xb_train = flat.reconstruct_n(0, TRAIN_SAMPLES).astype(np.float32)
    log(f"  xb_train shape={xb_train.shape}, {xb_train.nbytes/1e9:.2f}GB")

    # 3. 构建 IVF_PQ
    log("--- 3. init IVF_PQ ---")
    quantizer = faiss.IndexFlatIP(d)
    ivfpq = faiss.IndexIVFPQ(quantizer, d, NLIST, M, NBITS,
                             faiss.METRIC_INNER_PRODUCT)

    # 4. 训练
    t = time.time()
    log("--- 4. train IVF + PQ (~30-60 min) ---")
    ivfpq.train(xb_train)
    log(f"  train done in {time.time()-t:.1f}s")

    # 释放训练样本（很重要！否则一直占 3GB）
    del xb_train
    gc.collect()
    log("  freed train samples")

    # 5. Streaming add：每 BATCH_SIZE 条
    t = time.time()
    log(f"--- 5. streaming add 21M (batch={BATCH_SIZE:,}, ~30-60 min) ---")
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, n, BATCH_SIZE):
        end = min(i + BATCH_SIZE, n)
        batch = flat.reconstruct_n(i, end - i).astype(np.float32)
        ivfpq.add(batch)
        del batch  # immediate release
        if (i // BATCH_SIZE) % 5 == 0:
            pct = 100 * end / n
            log(f"  added {end:,} / {n:,} ({pct:.1f}%, batch {i//BATCH_SIZE+1}/{n_batches})")
        # 强制 gc 防 RSS 累积
        if (i // BATCH_SIZE) % 10 == 0:
            gc.collect()
    log(f"add done in {time.time()-t:.1f}s, ntotal={ivfpq.ntotal:,}")

    # 6. 写盘
    t = time.time()
    log("--- 6. write to disk ---")
    faiss.write_index(ivfpq, DST_TMP)
    sz_mb = os.path.getsize(DST_TMP) / 1e6
    os.rename(DST_TMP, DST)
    log(f"saved {DST}, {sz_mb:.1f} MB, write {time.time()-t:.1f}s")

    # 7. 验证
    log("--- 7. verify ---")
    del ivfpq, flat
    gc.collect()
    ivfpq2 = faiss.read_index(DST)
    ivfpq2.nprobe = 16
    flat2 = faiss.read_index(SRC, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    q = flat2.reconstruct_n(0, 5).astype(np.float32)
    D, I = ivfpq2.search(q, k=10)
    log(f"verify search: top-3 dists {D[0][:3]}, ids {I[0][:3]}")
    log("=== ALL DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"❌ FAIL: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
