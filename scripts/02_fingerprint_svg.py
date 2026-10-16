"""按标点指纹生成 SVG 热力图（供内联展示）。纯标准库。"""

LABS = [
    ("OpenAI", [
        ("gpt-5.5", 0.42, 0.11, 0.13, 0.09),
        ("gpt-5.6-luna", 0.51, 0.01, 0.37, 0.11),
        ("gpt-5.6-sol", 0.53, 0.00, 0.27, 0.15),
        ("gpt-5.6-terra", 0.47, 0.05, 0.23, 0.08),
        ("gpt-6-astra", 0.44, 0.09, 0.09, 0.07),
    ]),
    ("Google", [
        ("gemini-3.1-pro-preview", 0.06, 0.11, 0.00, 0.00),
        ("gemini-3.6-flash", 0.19, 0.21, 0.00, 0.00),
        ("gemini-3.7-flash", 0.31, 0.14, 0.01, 0.00),
        ("gemini-3.8-flash", 0.25, 0.11, 0.00, 0.02),
    ]),
    ("Anthropic", [
        ("claude-fable-5", 0.00, 0.60, 0.83, 0.00),
        ("claude-opus-4-6", 0.00, 0.57, 0.75, 0.00),
        ("claude-opus-4-8", 0.00, 0.60, 0.63, 0.00),
        ("claude-opus-5", 0.00, 0.63, 0.81, 0.00),
        ("claude-sonnet-5", 0.00, 0.73, 0.73, 0.00),
    ]),
    ("Qwen", [("qwen3.8-max", 0.29, 0.12, 0.03, 0.01)]),
    ("ZAI", [
        ("glm-5.3", 0.00, 0.58, 0.82, 0.00),
        ("glm-5.3-flash", 0.00, 0.67, 0.79, 0.00),
    ]),
    ("Moonshot AI", [("kimi-k3", 0.32, 0.23, 0.52, 0.09)]),
]

FILL = [
    (0.001, "#F1EFE8", "#888780"),
    (0.10, "#E6F1FB", "#042C53"),
    (0.25, "#B5D4F4", "#042C53"),
    (0.45, "#85B7EB", "#FFFFFF"),
    (0.65, "#378ADD", "#FFFFFF"),
    (1.01, "#185FA5", "#FFFFFF"),
]


def fill_for(v):
    for thr, bg, fg in FILL:
        if v < thr:
            return bg, fg
    return FILL[-1][1], FILL[-1][2]


X0, MODEL_W, CW = 40, 200, 100
HEAD_Y = 74
ROW_H, GAP = 22, 5
out = []
y = 96

for lab, models in LABS:
    for name, *vals in models:
        out.append(
            f'<text x="{X0 + 10}" y="{y + ROW_H / 2}" dominant-baseline="central" '
            f'font-family="var(--font-sans)" font-size="11" fill="#2C2C2A">{name}</text>'
        )
        for i, v in enumerate(vals):
            bg, fg = fill_for(v)
            cx = X0 + MODEL_W + i * CW
            out.append(
                f'<rect x="{cx + 1}" y="{y + 1}" width="{CW - 2}" height="{ROW_H - 2}" '
                f'rx="4" fill="{bg}"/>'
            )
            out.append(
                f'<text x="{cx + CW / 2}" y="{y + ROW_H / 2}" text-anchor="middle" '
                f'dominant-baseline="central" font-family="var(--font-sans)" '
                f'font-size="11" fill="{fg}">{v:.2f}</text>'
            )
        y += ROW_H
    out.append(
        f'<line x1="{X0}" y1="{y + GAP / 2}" x2="640" y2="{y + GAP / 2}" '
        f'stroke="#D3D1C7" stroke-width="0.5"/>'
    )
    y += GAP

H = int(y - GAP + 20)
labels = ["弯撇号 \u2019", "直撇号 '", "em dash \u2014", "弯双引 \u201c"]
head = [
    f'<text x="{X0 + MODEL_W + i * CW + CW / 2}" y="{HEAD_Y}" text-anchor="middle" '
    f'dominant-baseline="central" font-family="var(--font-sans)" font-size="11" '
    f'font-weight="500" fill="#2C2C2A">{lab}</text>'
    for i, lab in enumerate(labels)
]

print(
    f'<svg viewBox="0 0 680 {H}" width="100%" role="img" xmlns="http://www.w3.org/2000/svg">\n'
    f'<title>各模型的标点使用率指纹矩阵</title>\n'
    f'<desc>18 个模型在弯撇号、直撇号、em dash、弯双引号上的出现率，颜色越深比例越高。</desc>\n'
    f'<text x="{X0}" y="42" dominant-baseline="central" font-family="var(--font-sans)" '
    f'font-size="13" font-weight="500" fill="#2C2C2A">标点指纹矩阵（该字符在文本中出现的比例）</text>\n'
    + "\n".join(head) + "\n" + "\n".join(out) + "\n</svg>"
)
