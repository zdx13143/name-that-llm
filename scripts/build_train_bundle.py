"""构建「微调编码器」的 Kaggle 内核。

和数据载体上的区别：
- 外部语料用**公开数据集挂载**（olegzholobov/name-that-llm-response-corpus），
  不需要我们上传任何东西（本机上传接口被代理阻断）
- 模型用官方 Kaggle Model 挂载
- 同样是自包含单文件（script 内核只上传 code_file）

用法：
    python scripts/build_train_bundle.py
    kaggle kernels push -p kaggle/train
"""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "kaggle" / "train"

KERNEL_ID = "zdx13143/ntl-finetune"

EMBED_FILES = [
    "src/common.py",
    "src/assign.py",
    "src/train_clf.py",
]

RUNNER = '''
import base64
import pathlib
import sys
import tempfile

_work = pathlib.Path(tempfile.mkdtemp(prefix="ntl_train_"))
for _name, _b64 in _FILES.items():
    _dst = _work / _name
    _dst.parent.mkdir(parents=True, exist_ok=True)
    _dst.write_bytes(base64.b64decode(_b64))

sys.path.insert(0, str(_work / "src"))

from train_clf import main as _run

sys.argv = ["train_clf", "--out", "submission.parquet"] + {extra}
_run()
'''

# 冒烟模式：只跑第 1 折但跑满 4 轮，用来判断微调到底能不能超过融合模型的 0.3815
QUICK_ARGS = '["--only-fold", "1", "--epochs", "4"]'
FULL_ARGS = "[]"


def main() -> None:
    import sys as _sys

    quick = "--quick" in _sys.argv

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)

    payload = {rel: base64.b64encode((ROOT / rel).read_bytes()).decode("ascii")
               for rel in EMBED_FILES}
    body = ",\n".join(f'    "{k}": "{v}"' for k, v in payload.items())
    runner = RUNNER.replace("{extra}", QUICK_ARGS if quick else FULL_ARGS)
    (BUNDLE / "main.py").write_text(
        "#!/usr/bin/env python3\n_FILES = {\n" + body + "\n}\n" + runner,
        encoding="utf-8",
    )

    meta = {
        "id": KERNEL_ID,
        "title": "ntl-finetune",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        # 公开外部语料，直接挂载，无需上传
        "dataset_sources": ["olegzholobov/name-that-llm-response-corpus"],
        "model_sources": ["answer-ai/modernbert/transformers/base/2"],
        "competition_sources": ["name-that-llm"],
        "kernel_sources": [],
    }
    (BUNDLE / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"已生成 {BUNDLE / 'main.py'}（内嵌 {len(EMBED_FILES)} 个文件）")
    print(f"  kaggle kernels push -p {BUNDLE}")


if __name__ == "__main__":
    main()
