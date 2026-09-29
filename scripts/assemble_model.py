#!/usr/bin/env python3
"""Re-assemble the Qwen3-8B GGUF from the <95MB parts stored in git.

GitHub caps single files at 100 MB, so the model is shipped as
llm/qwen3-8b-parts/part-00000 ... part-000NN plus SHA256SUMS.txt.
This script concatenates the parts back into llm/qwen3-8b-q4km.gguf
and (optionally) verifies checksums.

Run:  python3 scripts/assemble_model.py [--skip-verify]
"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trader import config as C  # noqa: E402


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    parts_dir = C.MODEL_PARTS_DIR
    out_path = C.MODEL_GGUF
    sums_path = os.path.join(parts_dir, "SHA256SUMS.txt")

    if not os.path.isdir(parts_dir):
        print(f"[model] parts dir not found: {parts_dir}")
        return 1
    parts = sorted(p for p in os.listdir(parts_dir)
                   if p.startswith("part-") and not p.endswith(".sha256"))
    if not parts:
        print(f"[model] no part files in {parts_dir}")
        return 1

    expected = {}
    full_sha = None
    if os.path.exists(sums_path):
        with open(sums_path) as f:
            for ln in f:
                ln = ln.split()
                if len(ln) == 2:
                    if ln[1] not in ("qwen3-8b-q4km.gguf", "FULL"):
                        expected[ln[1]] = ln[0]
                    else:
                        full_sha = ln[0]

    if os.path.exists(out_path):
        try:
            size = os.path.getsize(out_path)
            done = size > 0
            if done and not args.skip_verify and full_sha:
                print(f"[model] {out_path} exists ({size:,} bytes) - verifying ...")
                if sha256_file(out_path) == full_sha:
                    print("[model] OK - already assembled and verified")
                    return 0
                print("[model] checksum mismatch - rebuilding")
            elif done:
                print(f"[model] {out_path} exists ({size:,} bytes) - skip")
                return 0
        except OSError:
            pass

    print(f"[model] concatenating {len(parts)} parts -> {out_path}")
    tmp = out_path + ".tmp"
    total = 0
    with open(tmp, "wb") as out:
        for i, p in enumerate(parts):
            path = os.path.join(parts_dir, p)
            with open(path, "rb") as f:
                while True:
                    b = f.read(1 << 24)
                    if not b:
                        break
                    out.write(b)
                    total += len(b)
            if i % 10 == 0 or i == len(parts) - 1:
                print(f"  [{i+1}/{len(parts)}] {total:,} bytes", flush=True)
    os.replace(tmp, out_path)
    print(f"[model] wrote {total:,} bytes")

    if not args.skip_verify:
        if expected:
            bad = 0
            for p in parts:
                if p in expected:
                    got = sha256_file(os.path.join(parts_dir, p))
                    if got != expected[p]:
                        print(f"  ! part checksum mismatch: {p}")
                        bad += 1
            print(f"[model] part checksums: {len(parts)-bad}/{len(parts)} OK")
            if bad:
                return 1
        if full_sha:
            print("[model] verifying full file sha256 ...")
            if sha256_file(out_path) == full_sha:
                print("[model] full-file checksum OK")
            else:
                print("[model] FULL-FILE CHECKSUM MISMATCH")
                return 1
    print("[model] ready:", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
