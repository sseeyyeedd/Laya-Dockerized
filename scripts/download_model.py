"""Pre-download the Laya checkpoint so the server runs fully offline.

Usage:
    python scripts/download_model.py                 # default endpoint
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_model.py

Honours HF_ENDPOINT for mirrors and resumes partially downloaded files.
The cache lands in HF_HOME (default: <repo>/.hf-cache).
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Keep the model cache inside the project so the venv and Docker builds are
# reproducible; override with a real HF_HOME if you want to share the cache.
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf-cache"))

from huggingface_hub import snapshot_download  # noqa: E402

REPO_ID = "convaiinnovations/laya"
SUBFOLDER = "multilingual"


def main() -> int:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"endpoint : {endpoint}")
    print(f"repo     : {REPO_ID} @ {SUBFOLDER}")
    print(f"cache    : {os.environ['HF_HOME']}")

    try:
        path = snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=[f"{SUBFOLDER}/*"],
            max_workers=4,
        )
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        print(
            "hint: if huggingface.co is unreachable, retry through a mirror:\n"
            "  HF_ENDPOINT=https://hf-mirror.com python scripts/download_model.py",
            file=sys.stderr,
        )
        return 1

    print(f"downloaded to: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
