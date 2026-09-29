"""Refresh docker-model/ (the build-context model staging dir) from .hf-cache.

Docker cannot follow the HF cache's symlinks on Windows, so we copy with
dereference and drop the redundant blobs/ dir. Run before `docker build` if
the cache changed:

    python scripts/stage_model.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / ".hf-cache" / "hub" / "models--convaiinnovations--laya"
DST = REPO_ROOT / "docker-model"

model_refs = (SRC / "refs" / "main")


def main() -> int:
    if not model_refs.exists():
        print(".hf-cache has no models--convaiinnovations--laya yet.", file=sys.stderr)
        print("run: python scripts/download_model.py", file=sys.stderr)
        return 1

    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir()

    # cp -rL dereferences symlinks; shutil.copytree(symlinks=False) does too,
    # but Git Bash cp is what reliably produces plain files on Windows here.
    subprocess.run(["cp", "-rL", str(SRC), str(DST)], check=True)
    blobs = DST / "models--convaiinnovations--laya" / "blobs"
    if blobs.exists():
        shutil.rmtree(blobs)

    size = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file())
    print(f"staged {size / 1e6:.0f} MB into {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
