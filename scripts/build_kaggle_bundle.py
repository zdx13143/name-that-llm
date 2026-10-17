"""把推理所需文件打包成可以直接 push 的 Kaggle 内核目录。

关键约束
--------
Kaggle 的 **script 类型内核只上传 `code_file` 一个文件**，同目录的其他 .py
不会进到运行环境里。所以这里生成一个**自包含的 main.py**：把各模块源码
base64 内嵌进去，运行时还原到临时目录再 import。

（不用字符串直接内嵌是因为源码里有三引号，容易把拼接写坏；base64 绝对不会错。）

生成的目录（kaggle/ 已被 .gitignore 排除）：

    kaggle/submit/
        kernel-metadata.json
        main.py            ← 唯一被上传的文件，自包含
        <其余文件仅本地留档>

用法：
    python scripts/build_kaggle_bundle.py
    kaggle kernels push -p kaggle/submit
"""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "kaggle" / "submit"

KERNEL_ID = "zdx13143/name-that-llm-submit"

# 要内嵌进 main.py 的文件（相对仓库根）
EMBED_FILES = [
    "src/common.py",
    "src/features.py",
    "src/cv.py",
    "src/run_sweep.py",
    "src/embeddings.py",
    "src/inference.py",
    "configs/blend_v1.json",
]

RUNNER = '''
import base64
import pathlib
import sys
import tempfile

_work = pathlib.Path(tempfile.mkdtemp(prefix="ntl_"))
for _name, _b64 in _FILES.items():
    _dst = _work / _name
    _dst.parent.mkdir(parents=True, exist_ok=True)
    _dst.write_bytes(base64.b64decode(_b64))

sys.path.insert(0, str(_work / "src"))

from inference import main as _run

sys.argv = ["inference", "--out", "submission.parquet"]
_run()
'''


def build_main_py() -> str:
    payload = {}
    for rel in EMBED_FILES:
        src = (ROOT / rel).read_bytes()
        # 保留完整相对路径：src/xxx.py -> _work/src/xxx.py，
        # configs/xxx.json -> _work/configs/xxx.json。
        # 丢前缀会让 sys.path 与 find_config 的定位全错。
        payload[rel] = base64.b64encode(src).decode("ascii")

    body = ",\n".join(f'    "{k}": "{v}"' for k, v in payload.items())
    return "#!/usr/bin/env python3\n_FILES = {\n" + body + "\n}\n" + RUNNER


def main() -> None:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)

    main_src = build_main_py()
    (BUNDLE / "main.py").write_text(main_src, encoding="utf-8")

    kernel_meta = {
        "id": KERNEL_ID,
        "title": "name-that-llm-submit",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        # 探针内核已确认 GPU 镜像自带 torch 2.11+cu128 / transformers 5.16.1 / T4
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [],
        # 官方 Kaggle Model（Apache 2.0）。用 transformers 框架的变体，
        # 里面是完整的 HF 仓库文件，AutoModel/AutoTokenizer 可直接加载。
        "model_sources": [
            "answer-ai/modernbert/transformers/base/2",
            "answer-ai/modernbert/transformers/large/2",
        ],
        "competition_sources": ["name-that-llm"],
        "kernel_sources": [],
    }
    (BUNDLE / "kernel-metadata.json").write_text(
        json.dumps(kernel_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"已生成自包含内核：{BUNDLE / 'main.py'}"
          f"（{len(main_src) / 1024:.0f} KB，内嵌 {len(EMBED_FILES)} 个文件）")
    print("\n下一步：")
    print(f"  kaggle kernels push -p {BUNDLE}")


if __name__ == "__main__":
    main()
