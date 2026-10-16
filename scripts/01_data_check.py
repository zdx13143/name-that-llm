"""快速数据体检 —— 不依赖 pandas，用纯标准库。"""
import csv
import re
from collections import Counter, defaultdict

TRAIN = "data/raw/train.csv"

rows = []
with open(TRAIN, encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

print("=" * 60)
print(f"总行数: {len(rows)}")
print(f"字段: {list(rows[0].keys())}")

# --- 标签分布 ---
models = Counter(r["model"] for r in rows)
print(f"\n唯一模型数: {len(models)}")
for m, c in sorted(models.items(), key=lambda x: -x[1]):
    print(f"  {m:<28} {c:>5}")

# --- prompt 结构 ---
prompts = Counter(r["prompt_id"] for r in rows)
print(f"\n唯一 prompt 数: {len(prompts)}")
dist = Counter(prompts.values())
print("每个 prompt 下的文本条数分布 (条数: prompt个数):")
for k in sorted(dist):
    print(f"  {k:>3} 条 -> {dist[k]:>4} 个 prompt")

# --- 每 prompt 的模型覆盖 ---
by_prompt = defaultdict(set)
for r in rows:
    by_prompt[r["prompt_id"]].add(r["model"])
missing = {p: sorted(set(models) - ms) for p, ms in by_prompt.items() if len(ms) < len(models)}
print(f"\n模型覆盖不全的 prompt 数: {len(missing)}")
for p, miss in list(missing.items())[:5]:
    print(f"  {p}: 缺 {miss}")

# --- 长度 ---
def wc(t):
    return len(t.split())

lens = [wc(r["text"]) for r in rows]
lens_sorted = sorted(lens)
n = len(lens_sorted)
print(f"\n词数: min={min(lens)} p25={lens_sorted[n//4]} median={lens_sorted[n//2]} "
      f"p75={lens_sorted[3*n//4]} max={max(lens)} mean={sum(lens)/n:.1f}")

# --- 换行 ---
multi = sum(1 for r in rows if "\n" in r["text"].strip())
print(f"含换行的文本: {multi}")

# --- 标点指纹 (核心信号) ---
print("\n" + "=" * 60)
print("标点指纹 (每模型: 弯撇号' / 直撇号' / em dash / 弯引号 出现过的文本占比)")
print(f"{'model':<28} {'弯撇':>6} {'直撇':>6} {'emdash':>7} {'弯双引':>7} {'平均词数':>8}")
punct = {}
for m in sorted(models):
    sub = [r["text"] for r in rows if r["model"] == m]
    curly_ap = sum(1 for t in sub if "\u2019" in t) / len(sub)
    straight_ap = sum(1 for t in sub if "'" in t) / len(sub)
    emdash = sum(1 for t in sub if "\u2014" in t) / len(sub)
    curly_q = sum(1 for t in sub if "\u201c" in t) / len(sub)
    avg_w = sum(wc(t) for t in sub) / len(sub)
    punct[m] = (curly_ap, straight_ap, emdash, curly_q, avg_w)
    print(f"{m:<28} {curly_ap:>6.2f} {straight_ap:>6.2f} {emdash:>7.2f} {curly_q:>7.2f} {avg_w:>8.1f}")

# --- 字符集里有哪些非 ASCII ---
nonascii = Counter()
for r in rows:
    for ch in r["text"]:
        if ord(ch) > 127:
            nonascii[ch] += 1
print("\n非 ASCII 字符 Top 15:")
for ch, c in nonascii.most_common(15):
    print(f"  U+{ord(ch):04X} {ch!r:<8} {c:>7}")
