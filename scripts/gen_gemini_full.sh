#!/usr/bin/env bash
# 全量：480 题 × 3 个 Gemini flash，走 Clash 代理（本机直连被墙）
set -euo pipefail
cd "$(dirname "$0")/.."
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
unset HTTPS_PROXY HTTP_PROXY
PY="${PY:-C:/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe}"
# 并发 12 × 单条约 71s ≈ 10 请求/分钟，稳在免费层 15 RPM 之下
"$PY" src/generate_synthetic.py --config configs/gemini.json \
  --out data/synthetic/gemini_flash.csv \
  --max-workers 12 --min-interval 0
