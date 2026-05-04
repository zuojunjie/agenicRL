#!/usr/bin/env python3
"""Build a numpy int64 array of byte offsets for each line in a jsonl file.

Output: <jsonl>.offsets.npy — array shape (N+1,) where offsets[i] = byte
offset of line i and offsets[N] = file size. Line i = bytes[offsets[i]:offsets[i+1]].

Memory: O(N). For 21M lines this is ~168MB.
Time:   bound by sequential disk read of the jsonl (1 pass, no parsing).
"""
import argparse
import os
import sys
import time

import numpy as np


def build(path: str) -> str:
    out_path = path + ".offsets.npy"
    if os.path.exists(out_path):
        print(f"[skip] {out_path} exists; delete if you want a rebuild")
        return out_path

    file_size = os.path.getsize(path)
    print(f"[build] scanning {path} ({file_size/1e9:.1f} GB)...", flush=True)

    offsets = [0]
    bytes_read = 0
    last_print = time.time()
    t0 = time.time()
    with open(path, "rb") as f:
        # Read in 64MB chunks; track newline positions for offsets.
        CHUNK = 64 * 1024 * 1024
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            base = bytes_read
            # find newline indices in chunk
            i = 0
            while True:
                j = chunk.find(b"\n", i)
                if j < 0:
                    break
                # next line starts after this newline
                offsets.append(base + j + 1)
                i = j + 1
            bytes_read += len(chunk)
            now = time.time()
            if now - last_print > 5:
                rate = bytes_read / (now - t0) / 1e6
                pct = bytes_read / file_size * 100
                print(f"  {pct:5.1f}%  {bytes_read/1e9:5.2f}/{file_size/1e9:.2f} GB  {rate:.0f} MB/s  N={len(offsets)-1}", flush=True)
                last_print = now

    # Last entry should equal file_size (handles file with or without trailing newline)
    if offsets[-1] != file_size:
        offsets.append(file_size)

    arr = np.asarray(offsets, dtype=np.int64)
    np.save(out_path, arr)
    print(f"[done] {len(arr)-1} lines, offsets file: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB) in {time.time()-t0:.1f}s")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl_path")
    args = ap.parse_args()
    build(args.jsonl_path)
