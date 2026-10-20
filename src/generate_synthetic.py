"""用 18 个目标模型的 API 批量生成带标签文本，作为外训练数据。

为什么需要这一步
----------------
本地验证与真实榜分一致（CV 0.3882 vs LB 0.381），说明模型没作弊也没过拟合，
就是**数据不够**：每类只有 150 条，要在 150 个全新 prompt 上区分 18 个模型。
榜首 0.59 最合理的解释就是他们拉开了数据量。

规则依据
--------
官方 Rules 6.a：外部数据必须「对所有参赛者公开且平等可获取、无成本」，或满足
6.b 的合理性标准；6.b 明确写了「The use of external data and models is acceptable
unless specifically prohibited by the Host」，并举例小额的模型订阅费用是可接受的。
所以用 API 自造训练数据是被预案过的合法操作。
**但生成完成后必须公开**（Kaggle 数据集 / discussion / 公开仓库），否则有争议。

用法
----
    # 1) 先小规模试跑，确认每个模型都能通
    python src/generate_synthetic.py --config configs/models.json \
        --out data/synthetic/trial.csv --limit-per-model 8 --dry-run

    # 2) 正式生成（可中断，重跑会自动续）
    python src/generate_synthetic.py --config configs/models.json \
        --out data/synthetic/synthetic.csv --samples-per-prompt 2

    # 3) 只看数据质量
    python src/generate_synthetic.py --config configs/models.json \
        --out data/synthetic/synthetic.csv --report-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_prompts import build_prompts  # noqa: E402

COLUMNS = ["id", "prompt_id", "text", "model"]
_write_lock = threading.Lock()
_rate_lock = threading.Lock()
_last_call = [0.0]


def _throttle(min_interval: float) -> None:
    """全局限速：保证相邻请求至少间隔 min_interval 秒（多线程共享）。"""
    if min_interval <= 0:
        return
    with _rate_lock:
        now = time.time()
        wait = _last_call[0] + min_interval - now
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()

# 支持从仓库根目录的 .env 读 key，省得去配系统环境变量。
# .env 已在 .gitignore 与推送忽略列表里，不会入库。
ENV_KEYS = (
    "OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
    "DASHSCOPE_API_KEY", "ZHIPU_API_KEY", "MOONSHOT_API_KEY",
)


def load_dotenv() -> list[str]:
    """把 .env 里的键值塞进 os.environ（不覆盖已有变量）。返回加载了哪些键。"""
    loaded: list[str] = []
    here = Path(__file__).resolve().parents[1]
    for cand in (here / ".env", Path.cwd() / ".env"):
        if not cand.exists():
            continue
        for raw in cand.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and not os.environ.get(k):
                os.environ[k] = v
                loaded.append(k)
    return loaded


def make_id(prompt_id: str, model: str, sample: int) -> str:
    raw = f"{prompt_id}|{model}|{sample}".encode()
    return hashlib.md5(raw).hexdigest()[:12]


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    providers = cfg["providers"]
    models = cfg["models"]

    active, skipped = {}, {}
    for label, spec in models.items():
        prov = providers.get(spec["provider"])
        if prov is None:
            skipped[label] = f"未知 provider {spec['provider']}"
            continue
        key = os.environ.get(prov.get("api_key_env", ""), "").strip()
        if not key:
            skipped[label] = f"缺少环境变量 {prov.get('api_key_env')}"
            continue
        active[label] = {
            "base_url": prov["base_url"].rstrip("/"),
            "api_key": key,
            "model_id": spec.get("model_id") or label,
            "extra": spec.get("extra", {}),
        }
    return {"active": active, "skipped": skipped}


def call_once(spec: dict, prompt: str, timeout: int = 120,
              max_tokens: int = 450, temperature: float = 1.0,
              retries: int = 4, min_interval: float = 0.0) -> str:
    url = f"{spec['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {spec['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": spec["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    payload.update(spec.get("extra", {}))

    last_err = ""
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"] or ""
                return text.strip()
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {str(exc)[:200]}"
        time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(last_err)


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=COLUMNS)
    header = not path.exists() or path.stat().st_size == 0
    with _write_lock:
        df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")


def report(path: Path) -> None:
    df = pd.read_csv(path)
    df["n_words"] = df["text"].astype(str).str.split().str.len()
    print(f"\n总行数 {len(df)}")
    print(f"唯一 id {df['id'].nunique()}  |  prompt {df['prompt_id'].nunique()}  "
          f"|  模型 {df['model'].nunique()}")
    print("\n每个模型的条数 / 中位词数：")
    agg = df.groupby("model").agg(n=("id", "size"), med_words=("n_words", "median"))
    print(agg.sort_values("n").to_string(float_format=lambda v: f"{v:.0f}"))
    print(f"\n词数分布: min={df.n_words.min()} p25={df.n_words.quantile(.25):.0f} "
          f"median={df.n_words.median():.0f} p75={df.n_words.quantile(.75):.0f} "
          f"max={df.n_words.max()}")
    short = (df.n_words < 30).sum()
    print(f"过短（<30 词）{short} 条  过长（>400 词）{(df.n_words > 400).sum()} 条")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models.json")
    ap.add_argument("--out", default="data/synthetic/synthetic.csv")
    ap.add_argument("--samples-per-prompt", type=int, default=1)
    ap.add_argument("--limit-per-model", type=int, default=0,
                    help="每个模型最多生成多少条（试跑用）")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--min-interval", type=float, default=0.0,
                    help="两次请求之间的最小间隔（秒）。Gemini 免费层 15 RPM 建议 4.5")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        report(out)
        return

    loaded = load_dotenv()
    if loaded:
        print(f"从 .env 读到 {len(loaded)} 个 key：{', '.join(sorted(loaded))}\n")

    cfg = load_config(Path(args.config))
    active, skipped = cfg["active"], cfg["skipped"]
    print(f"可用模型 {len(active)} / 18")
    if active:
        for k in sorted(active):
            print(f"  ✓ {k:<24} -> {active[k]['model_id']}")
    if skipped:
        print(f"\n跳过 {len(skipped)} 个：")
        for k, why in sorted(skipped.items()):
            print(f"  ✗ {k:<24} {why}")
    if not active:
        sys.exit("\n没有任何可用模型，检查 configs/models.json 和 API key 环境变量")

    prompts = build_prompts()
    print(f"\nprompt 库 {len(prompts)} 条 × 每 prompt {args.samples_per_prompt} 次采样")

    # 断点续跑：已经生成过的 (prompt, model, sample) 直接跳过
    done: set[tuple[str, str, int]] = set()
    if out.exists():
        old = pd.read_csv(out, usecols=["prompt_id", "model"])
        # 无法从旧数据反推 sample 序号，这里按 (prompt, model) 计数保守跳过
        counts = old.groupby(["prompt_id", "model"]).size()
        for (pid, model), n in counts.items():
            for s in range(int(n)):
                done.add((pid, model, s))
        print(f"已有 {len(old)} 行，按 (prompt, model, sample) 续跑")

    tasks = []
    for label in sorted(active):
        per_model = 0
        for p in prompts:
            for s in range(args.samples_per_prompt):
                if (p["prompt_id"], label, s) in done:
                    continue
                if args.limit_per_model and per_model >= args.limit_per_model:
                    break
                tasks.append((label, p["prompt_id"], p["prompt_text"], s))
                per_model += 1
            if args.limit_per_model and per_model >= args.limit_per_model:
                break
    random.shuffle(tasks)
    print(f"待生成 {len(tasks)} 条\n")

    if args.dry_run:
        for label, pid, text, s in tasks[:10]:
            print(f"  [{label}] {pid}#{s}  {text[:70]}...")
        print(f"\n（dry-run 未发请求；共 {len(tasks)} 条待生成）")
        return

    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {}
        for label, pid, text, s in tasks:
            fut = pool.submit(
                call_once, active[label], text, 120,
                args.max_tokens, args.temperature, 4, args.min_interval,
            )
            futures[fut] = (label, pid, s)

        for i, fut in enumerate(as_completed(futures), 1):
            label, pid, s = futures[fut]
            try:
                text = fut.result()
            except Exception as exc:  # noqa: BLE001
                fail += 1
                if fail <= 10:
                    print(f"  ✗ {label} {pid}#{s}: {str(exc)[:140]}")
                continue
            if not text or len(text.split()) < 10:
                fail += 1
                continue
            append_rows(out, [{
                "id": make_id(pid, label, s),
                "prompt_id": pid,
                "text": " ".join(text.split()),
                "model": label,
            }])
            ok += 1
            if i % 100 == 0:
                rate = i / max(time.time() - t0, 1e-6)
                print(f"  进度 {i}/{len(tasks)}  成功 {ok}  失败 {fail}  "
                      f"{rate:.1f} 条/秒")

    print(f"\n完成：成功 {ok}，失败 {fail}，耗时 {time.time() - t0:.0f}s")
    if out.exists():
        report(out)


if __name__ == "__main__":
    main()
