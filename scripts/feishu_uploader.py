#!/usr/bin/env python3
"""[DEPRECATED — kept for educational reference]

Push local markdown files into Feishu cloud docs via raw OpenAPI.

⚠️  Use `lark-cli` + the `lark-doc` skill instead. It is strictly more
    capable (XML/markdown both, fetch, update, search, media, ...) and
    creates docs under USER identity (so the user can edit them later
    via any tool). See docs/feishu-export/00-master.md "维护元信息"
    for the recommended recipe.

Why this script exists at all:
    - subAgent 1 (worktree sweet-archimedes-7376d4) wrote this on
      2026-04-26 before discovering ~/.claude/skills/lark-doc/.
    - First-batch docs uploaded by this script were bot-owned and
      could not be edited by the user via lark-cli — all 6 had to be
      re-uploaded via lark-cli the same evening.
    - Kept as: a 60-line reference for the Feishu OpenAPI 3-step flow
      (auth/v3 → drive/v1/medias/upload_all → drive/v1/import_tasks).

Reads credentials from ~/.config/feishu/credentials.json:
    {"app_id": "cli_...", "app_secret": "..."}

Caches tenant_access_token at /tmp/feishu_token.json (TTL 7200s).

Pure stdlib (urllib + json) — no external deps required.

Examples (still functional, not recommended):
    python feishu_uploader.py upload notes/agent-loop.md
    python feishu_uploader.py upload-dir docs/feishu-export/
    python feishu_uploader.py whoami
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import urllib.error
import urllib.parse
import urllib.request

API = "https://open.feishu.cn/open-apis"
CREDS_PATH = Path.home() / ".config" / "feishu" / "credentials.json"
TOKEN_CACHE = Path("/tmp/feishu_token.json")


# ─────────────────────────── HTTP plumbing ───────────────────────────

def _http(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: Any = None,
    form: dict[str, str] | None = None,
    files: list[tuple[str, str, bytes, str]] | None = None,
    timeout: int = 60,
) -> dict:
    """Minimal HTTP helper. files = list of (field, filename, bytes, mime)."""
    h = dict(headers or {})
    data: bytes | None = None
    if files is not None:
        boundary = uuid.uuid4().hex
        body = b""
        for k, v in (form or {}).items():
            body += (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
            ).encode()
        for field, filename, content, mime in files:
            body += (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
                f"Content-Type: {mime}\r\n\r\n"
            ).encode()
            body += content + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        h["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        data = body
    elif json_body is not None:
        h["Content-Type"] = "application/json; charset=utf-8"
        data = json.dumps(json_body).encode()
    elif form is not None:
        h["Content-Type"] = "application/x-www-form-urlencoded"
        data = urllib.parse.urlencode(form).encode()
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode())
        except Exception:
            return {"code": e.code, "msg": str(e), "_raw_http_error": True}


# ─────────────────────────── auth ───────────────────────────

def _load_creds() -> dict:
    if not CREDS_PATH.exists():
        sys.exit(
            f"❌ no credentials at {CREDS_PATH}.\n"
            'put {"app_id": "cli_...", "app_secret": "..."} there with chmod 600.'
        )
    return json.loads(CREDS_PATH.read_text())


def get_token() -> str:
    """Get tenant_access_token, using filesystem cache if still fresh."""
    if TOKEN_CACHE.exists():
        cached = json.loads(TOKEN_CACHE.read_text())
        if cached.get("expire_at", 0) > time.time() + 60:
            return cached["token"]
    creds = _load_creds()
    r = _http(
        "POST",
        f"{API}/auth/v3/tenant_access_token/internal",
        json_body={"app_id": creds["app_id"], "app_secret": creds["app_secret"]},
    )
    if r.get("code") != 0:
        sys.exit(f"❌ token request failed: {r}")
    tok = r["tenant_access_token"]
    TOKEN_CACHE.write_text(
        json.dumps({"token": tok, "expire_at": time.time() + r["expire"]})
    )
    return tok


# ─────────────────────────── upload pipeline ───────────────────────────

def _upload_media(path: Path, *, token: str) -> str:
    """Step 1: upload .md as import-targeted media. Returns file_token."""
    content = path.read_bytes()
    r = _http(
        "POST",
        f"{API}/drive/v1/medias/upload_all",
        headers={"Authorization": f"Bearer {token}"},
        form={
            "file_name": path.name,
            "parent_type": "ccm_import_open",
            "parent_node": "",
            "size": str(len(content)),
            "extra": json.dumps({"obj_type": "docx", "file_extension": "md"}),
        },
        files=[("file", path.name, content, "text/markdown")],
    )
    if r.get("code") != 0:
        raise RuntimeError(f"upload_all failed: {r}")
    return r["data"]["file_token"]


def _create_import(file_token: str, *, name: str, folder: str, token: str) -> str:
    """Step 2: trigger md→docx conversion. Returns ticket."""
    r = _http(
        "POST",
        f"{API}/drive/v1/import_tasks",
        headers={"Authorization": f"Bearer {token}"},
        json_body={
            "file_extension": "md",
            "file_token": file_token,
            "type": "docx",
            "file_name": name,
            "point": {
                "mount_type": 1,  # 1 = explorer / 我的空间
                "mount_key": folder,
            },
        },
    )
    if r.get("code") != 0:
        raise RuntimeError(f"import_tasks failed: {r}")
    return r["data"]["ticket"]


def _poll_import(ticket: str, *, token: str, timeout_s: int = 60) -> dict:
    """Step 3: poll until done. Returns final result dict with .token, .url."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        time.sleep(1.5)
        r = _http(
            "GET",
            f"{API}/drive/v1/import_tasks/{ticket}",
            headers={"Authorization": f"Bearer {token}"},
        )
        last = r
        if r.get("code") != 0:
            raise RuntimeError(f"poll failed: {r}")
        result = r.get("data", {}).get("result", {})
        status = result.get("job_status")
        if status == 0:
            return result
        if status not in (None, 1, 2):  # 1=pending, 2=processing
            raise RuntimeError(f"import job failed: {result}")
    raise TimeoutError(f"import not done in {timeout_s}s, last={last}")


def upload_md(
    path: str | Path,
    *,
    folder: str = "",
    name: str | None = None,
) -> dict:
    """High-level: push one .md → return {url, token, name, source}."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() != ".md":
        raise ValueError(f"only .md supported, got {p.suffix}")
    title = name or p.stem
    tok = get_token()
    file_token = _upload_media(p, token=tok)
    ticket = _create_import(file_token, name=title, folder=folder, token=tok)
    result = _poll_import(ticket, token=tok)
    doc_token = result.get("token")
    return {
        "name": title,
        "source": str(p),
        "doc_token": doc_token,
        "url": result.get("url") or f"https://feishu.cn/docx/{doc_token}",
    }


# ─────────────────────────── CLI ───────────────────────────

def cmd_upload(args: argparse.Namespace) -> None:
    out = upload_md(args.path, folder=args.folder, name=args.name)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_upload_dir(args: argparse.Namespace) -> None:
    d = Path(args.dir)
    files = sorted(d.glob(args.pattern))
    if not files:
        sys.exit(f"no files matching {args.pattern} under {d}")
    results = []
    for f in files:
        print(f"→ {f.name}", flush=True)
        try:
            r = upload_md(f, folder=args.folder)
            print(f"  ✅ {r['url']}")
            results.append(r)
        except Exception as e:
            print(f"  ❌ {e}")
            results.append({"source": str(f), "error": str(e)})
    print()
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\nresults saved → {args.out}")


def cmd_whoami(_: argparse.Namespace) -> None:
    creds = _load_creds()
    print(f"app_id: {creds['app_id']}")
    print(f"app_secret: {creds['app_secret'][:6]}…(redacted)")
    tok = get_token()
    print(f"token (first 12): {tok[:12]}…")
    if TOKEN_CACHE.exists():
        c = json.loads(TOKEN_CACHE.read_text())
        ttl = int(c["expire_at"] - time.time())
        print(f"cache TTL: {ttl}s")


def main() -> None:
    p = argparse.ArgumentParser(prog="feishu_uploader")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("upload", help="upload one .md to Feishu")
    a.add_argument("path")
    a.add_argument("--folder", default="", help="folder_token; empty = My Space root")
    a.add_argument("--name", default=None, help="doc title; default = file stem")
    a.set_defaults(func=cmd_upload)

    b = sub.add_parser("upload-dir", help="upload all .md in a directory")
    b.add_argument("dir")
    b.add_argument("--pattern", default="*.md")
    b.add_argument("--folder", default="")
    b.add_argument("--out", default=None, help="save JSON results here")
    b.set_defaults(func=cmd_upload_dir)

    c = sub.add_parser("whoami", help="check auth + show creds summary")
    c.set_defaults(func=cmd_whoami)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
