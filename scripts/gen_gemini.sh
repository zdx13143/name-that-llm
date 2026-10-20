#!/usr/bin/env bash
# 走 Clash Verge 的本地代理生成 Gemini 数据（本机直连被墙，必须走 7897）
set -euo pipefail
cd "$(dirname "$0")/.."

# Clash Verge 的混合端口；如改过端口，改这里
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
unset HTTPS_PROXY HTTP_PROXY

PY="${PY:-C:/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe}"

echo "== 1/3 连通性检查 =="
code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 \
  "https://generativelanguage.googleapis.com/v1beta/models?key=${GEMINI_API_KEY:-none}")
echo "  models 端点 HTTP $code （200=key 有效，400/403=key 有问题）"

echo "== 2/3 小试 3 条 =="
"$PY" src/generate_synthetic.py --config configs/gemini.json \
  --out data/synthetic/gemini_trial.csv --limit-per-model 3 \
  --max-workers 1 --min-interval 4.5

echo "== 3/3 检查质量 =="
"$PY" - <<'PY'
import pandas as pd, pathlib
p = pathlib.Path("data/synthetic/gemini_trial.csv")
if not p.exists() or p.stat().st_size == 0:
    print("  没有产出，检查 key 和代理"); raise SystemExit(1)
d = pd.read_csv(p)
d["n_words"] = d["text"].astype(str).str.split().str.len()
print(d.groupby("model")["n_words"].describe()[["count","mean","min","max"]].to_string())
print("\n样例（前 160 字）:")
for _, r in d.head(3).iterrows():
    print(f"  [{r['model']}] {str(r['text'])[:160]}")
PY

echo ""
echo "小试通过后，跑全量："
echo "  bash scripts/gen_gemini_full.sh"
