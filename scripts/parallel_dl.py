"""
Parallel multi-range downloader using hf-mirror.com.
Saturates AutoDL's bandwidth via 8 concurrent range requests.
Resume-friendly: detects existing partial file and continues from there.
"""
import os, sys, time, requests, concurrent.futures
from pathlib import Path

URL_BASE = "https://hf-mirror.com"
OUT_DIR = Path("/root/autodl-tmp/data/wikipedia_index")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARALLELISM = 8

# 不要走 AutoDL 代理（实测 hf-mirror 直连更快）
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(k, None)


def get_size(url):
    r = requests.head(url, allow_redirects=True, timeout=30)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def download_range(url, start, end, out_path):
    headers = {"Range": f"bytes={start}-{end}"}
    backoff = 1
    while True:
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(out_path, "r+b") as f:
                    f.seek(start)
                    for chunk in r.iter_content(8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return end - start + 1
        except Exception as e:
            print(f"  retry [{start}..{end}]: {e}", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)


def download_one(repo_path, filename):
    url = f"{URL_BASE}/{repo_path}/resolve/main/{filename}"
    out = OUT_DIR / filename

    print(f"\n=== {filename} ===", flush=True)
    total = get_size(url)
    print(f"total: {total/1e9:.2f} GB", flush=True)

    if not out.exists():
        out.touch()
    cur = out.stat().st_size
    if cur >= total:
        print("already complete", flush=True)
        return

    # 把文件预扩到目标大小，便于 random offset 写入
    with open(out, "r+b") as f:
        f.truncate(total)

    remaining = total - cur
    print(f"resume from {cur/1e9:.2f} GB, remaining {remaining/1e9:.2f} GB", flush=True)

    chunk_sz = remaining // PARALLELISM
    chunks = []
    for i in range(PARALLELISM):
        s = cur + i * chunk_sz
        e = s + chunk_sz - 1 if i < PARALLELISM - 1 else total - 1
        chunks.append((s, e))

    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLELISM) as ex:
        futs = {ex.submit(download_range, url, s, e, out): (s, e) for s, e in chunks}
        for f in concurrent.futures.as_completed(futs):
            n = f.result()
            done += n
            el = time.time() - t0
            print(f"chunk done: total +{done/1e9:.2f}GB in {el:.0f}s = {done/el/1e6:.1f} MB/s", flush=True)

    print(f"{filename} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    print(f"started at {time.strftime('%H:%M:%S')}", flush=True)
    download_one("datasets/PeterJinGo/wiki-18-e5-index", "part_aa")
    download_one("datasets/PeterJinGo/wiki-18-e5-index", "part_ab")
    download_one("datasets/PeterJinGo/wiki-18-corpus", "wiki-18.jsonl.gz")
    print(f"\n=== ALL DONE at {time.strftime('%H:%M:%S')} ===", flush=True)
