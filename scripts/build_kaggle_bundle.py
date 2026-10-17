"""把推理所需文件打包成可以直接 push 的 Kaggle 内核目录。

生成的目录结构（全部落在 kaggle/ 下，已被 .gitignore 排除）：

    kaggle/submit/
        kernel-metadata.json
        main.py            ← 提交入口，调用 inference.main()
        inference.py       ← 从这里开始全是 src/ 的拷贝
        common.py  features.py  cv.py  run_sweep.py  embeddings.py
        configs/blend_v1.json

用法：
    python scripts/build_kaggle_bundle.py
    kaggle datasets create -p models            # 首次上传模型数据集
    kaggle kernels push -p kaggle/submit        # 推送内核并提交
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "kaggle" / "submit"

KERNEL_ID = "zdx13143/name-that-llm-submit"
DATASET_ID = "zdx13143/name-that-llm-models"

MAIN_PY = '''"""Kaggle 提交入口。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import main

if __name__ == "__main__":
    sys.argv = ["inference", "--out", "submission.parquet"]
    main()
'''

SRC_FILES = [
    "common.py",
    "features.py",
    "cv.py",
    "run_sweep.py",
    "embeddings.py",
    "inference.py",
]


def main() -> None:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    (BUNDLE / "configs").mkdir(parents=True)

    for name in SRC_FILES:
        shutil.copy2(ROOT / "src" / name, BUNDLE / name)
    shutil.copy2(ROOT / "configs" / "blend_v1.json", BUNDLE / "configs" / "blend_v1.json")
    (BUNDLE / "main.py").write_text(MAIN_PY, encoding="utf-8")

    kernel_meta = {
        "id": KERNEL_ID,
        "title": "name-that-llm-submit",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [DATASET_ID],
        "competition_sources": ["name-that-llm"],
        "kernel_sources": [],
    }
    (BUNDLE / "kernel-metadata.json").write_text(
        json.dumps(kernel_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 模型数据集：直接上传 models/ 目录，metadata 放进目录里即可
    ds_meta = {
        "title": "Name That LLM Models",
        "id": DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (ROOT / "models" / "dataset-metadata.json").write_text(
        json.dumps(ds_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"内核目录已生成：{BUNDLE}")
    for p in sorted(BUNDLE.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(BUNDLE)}")
    print(f"\n模型数据集元信息已写入 {ROOT / 'models' / 'dataset-metadata.json'}")
    print("\n下一步：")
    print(f"  kaggle datasets create -p {ROOT / 'models'}")
    print(f"  kaggle kernels push -p {BUNDLE}")


if __name__ == "__main__":
    main()
