"""把逐类召回率画成 SVG 横向条形图。

用法：
    python scripts/03_recall_chart.py outputs/blend_v2_oof.npy
用于快速定位「分数丢在哪一类」，比赛期间每次迭代都值得重跑一遍。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import LAB_OF, LABELS, LABEL2IDX  # noqa: E402

BLUE = "#378ADD"
RED = "#E24B4A"
GRAY = "#888780"


def main(oof_path: str) -> None:
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/raw/train.csv")
    y = np.array([LABEL2IDX[m] for m in df["model"]])
    pred = np.load(oof_path).argmax(axis=1)

    rows = []
    for i, m in enumerate(LABELS):
        sup = int((y == i).sum())
        ok = int(((y == i) & (pred == i)).sum())
        rows.append({"model": m, "lab": LAB_OF[m], "recall": ok / sup})
    r = pd.DataFrame(rows).sort_values("recall", ascending=False).reset_index(drop=True)
    mean = float(r["recall"].mean())

    X0, LABEL_W, BAR_W = 40, 190, 330
    Y0, ROW_H = 96, 22
    vmax = 0.80
    out = []

    # 均值参考线
    mx = X0 + LABEL_W + BAR_W * mean / vmax
    out.append(
        f'<line x1="{mx:.1f}" y1="{Y0 - 12}" x2="{mx:.1f}" y2="{Y0 + len(r) * ROW_H - 8}" '
        f'stroke="{GRAY}" stroke-width="0.5" stroke-dasharray="3 3"/>'
    )
    out.append(
        f'<text x="{mx + 4:.1f}" y="{Y0 - 18}" font-family="var(--font-sans)" '
        f'font-size="11" fill="{GRAY}">均值 {mean:.3f}</text>'
    )

    for i, row in r.iterrows():
        ypos = Y0 + i * ROW_H
        cy = ypos + ROW_H / 2 - 2
        w = max(BAR_W * row["recall"] / vmax, 1.5)
        color = BLUE if row["recall"] >= mean else RED
        out.append(
            f'<text x="{X0}" y="{cy:.1f}" dominant-baseline="central" '
            f'font-family="var(--font-sans)" font-size="11" fill="#2C2C2A">'
            f'{row["model"]}</text>'
        )
        out.append(
            f'<rect x="{X0 + LABEL_W}" y="{ypos}" width="{w:.1f}" height="{ROW_H - 6}" '
            f'rx="3" fill="{color}"/>'
        )
        out.append(
            f'<text x="{X0 + LABEL_W + w + 6:.1f}" y="{cy:.1f}" dominant-baseline="central" '
            f'font-family="var(--font-sans)" font-size="11" fill="{GRAY}">'
            f'{row["recall"]:.3f}</text>'
        )

    h = Y0 + len(r) * ROW_H + 8
    print(
        f'<svg viewBox="0 0 680 {h}" width="100%" role="img" '
        f'xmlns="http://www.w3.org/2000/svg">\n'
        f'<title>逐类召回率</title>\n'
        f'<desc>18 个模型各自被正确识别的比例，红色表示低于均值的短板类。</desc>\n'
        f'<text x="{X0}" y="46" dominant-baseline="central" '
        f'font-family="var(--font-sans)" font-size="13" font-weight="500" '
        f'fill="#2C2C2A">逐类召回率（balanced accuracy 就是这些数的平均）</text>\n'
        f'<text x="{X0}" y="68" dominant-baseline="central" '
        f'font-family="var(--font-sans)" font-size="11" fill="{GRAY}">'
        f'红色 = 拖后腿的类，提升空间几乎全在这些类上</text>\n'
        + "\n".join(out) + "\n</svg>"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/blend_v2_oof.npy")
