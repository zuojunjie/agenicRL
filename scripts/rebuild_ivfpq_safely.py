"""
IVF_PQ 索引重建（生产级别安全）— 保证不影响并行训练
设计:
  1. faiss-cpu only，0 GPU 抢占
  2. OMP threads 限 32（不抢 96 vCPU 全部）
  3. nice +19 最低 CPU 优先级
  4. 内存监控 + 进度日志
  5. 异常 abort
用法:
  # 主进程:
  taskset -c 16-95 nice -n 19 python rebuild_ivfpq_safely.py
  # 监控（另一终端）:
  watch -n 30 'cat /tmp/ivfpq_progress.log'
"""
import os
import sys
import time
import faiss
import numpy as np
import psutil
import json

# ---- 配置 ----
SRC = "/root/autodl-tmp/data/wikipedia_index/e5_Flat.index"
DST = "/root/autodl-tmp/data/wikipedia_index/e5_IVF_PQ.index"
DST_TMP = DST + ".tmp"
PROGRESS_LOG = "/tmp/ivfpq_progress.log"

NLIST = 4096           # IVF 簇数
M = 8                  # PQ 子量化器数（768 / 8 = 96 dim per sub）
NBITS = 8              # 每子空间 256 中心点
TRAIN_SAMPLES = 1_000_000  # 用 1M 样本训练 IVF + PQ

# OMP 线程数：限 32 不抢 96 vCPU
faiss.omp_set_num_threads(32)
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"


def log(msg):
    """每条日志同时写文件 + stdout"""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")


def get_self_status():
    p = psutil.Process()
    return {
        "rss_gb": p.memory_info().rss / 1e9,
        "cpu_pct": p.cpu_percent(0.1),
        "threads": p.num_threads(),
    }


def main():
    log(f"=== IVF_PQ 重建开始 ===")
    log(f"OMP threads: {os.environ['OMP_NUM_THREADS']}")
    log(f"nice: {os.nice(0)}")  # 当前 nice 值
    log(f"src: {SRC}")
    log(f"dst: {DST}")

    # 1. 读 Flat 索引
    t = time.time()
    log(f"--- 1. read Flat index (~40s, RSS +64GB) ---")
    flat = faiss.read_index(SRC)
    log(f"loaded: {type(flat).__name__}, n={flat.ntotal:,}, d={flat.d}, in {time.time()-t:.1f}s")
    log(f"status: {get_self_status()}")

    # 2. 提取所有向量到 numpy（用 reconstruct_n batch）
    t = time.time()
    log(f"--- 2. reconstruct_n 全量 (~5-10 min, +64GB) ---")
    n, d = flat.ntotal, flat.d
    xb = np.empty((n, d), dtype='float32')
    B = 1_000_000
    for i in range(0, n, B):
        end = min(i + B, n)
        xb[i:end] = flat.reconstruct_n(i, end - i)
        if (i // B) % 5 == 0:
            log(f"  reconstructed {i+B:,} / {n:,} ({100*(i+B)/n:.1f}%)")
    log(f"reconstruct done in {time.time()-t:.1f}s")

    # 释放 Flat（np 数组里有数据了）
    del flat
    import gc; gc.collect()
    log(f"after del flat: {get_self_status()}")

    # 3. 构建 IVF_PQ 索引（IP metric 与原 Flat 一致）
    log(f"--- 3. build IVF_PQ ---")
    log(f"  nlist={NLIST}, m={M}, nbits={NBITS}, metric=INNER_PRODUCT")
    quantizer = faiss.IndexFlatIP(d)
    ivfpq = faiss.IndexIVFPQ(quantizer, d, NLIST, M, NBITS,
                             faiss.METRIC_INNER_PRODUCT)

    # 4. 训练（用 1M 子集）
    t = time.time()
    log(f"--- 4. train IVF + PQ (1M samples, ~30-60 min) ---")
    rng = np.random.default_rng(42)
    train_idx = rng.choice(n, TRAIN_SAMPLES, replace=False)
    train_idx.sort()  # 顺序访问，cache 友好
    xb_train = xb[train_idx].copy()
    log(f"  training subset: {xb_train.shape} ({xb_train.nbytes/1e9:.1f} GB)")
    ivfpq.train(xb_train)
    log(f"train done in {time.time()-t:.1f}s")
    del xb_train

    # 5. 加全量
    t = time.time()
    log(f"--- 5. add 21M vectors (~30-60 min) ---")
    ivfpq.add(xb)
    log(f"add done in {time.time()-t:.1f}s, ntotal={ivfpq.ntotal:,}")

    # 6. 写盘
    t = time.time()
    log(f"--- 6. write to disk ---")
    faiss.write_index(ivfpq, DST_TMP)
    sz_mb = os.path.getsize(DST_TMP) / 1e6
    os.rename(DST_TMP, DST)
    log(f"saved {DST}, size {sz_mb:.1f} MB, write {time.time()-t:.1f}s")

    # 7. 验证（reload + 1 次 search 测试）
    log(f"--- 7. verify reload + search ---")
    ivfpq2 = faiss.read_index(DST)
    ivfpq2.nprobe = 16  # 检索时探索 16 个簇
    q = xb[0:5]  # 用前 5 条做 query
    D, I = ivfpq2.search(q, k=10)
    log(f"verify search: top-10 distances {D[0][:3]}, indices {I[0][:3]}")

    log(f"=== ALL DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"❌ FAIL: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
