"""Kaggle 提交入口。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import main

if __name__ == "__main__":
    sys.argv = ["inference", "--out", "submission.parquet"]
    main()
