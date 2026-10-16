"""公共常量、路径与数据读取。

所有模块统一从这里取路径和标签，避免各处硬编码。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"

# 官方要求的 18 个合法标签，顺序与比赛页一致，一字不能差。
LABELS: list[str] = [
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "qwen3.8-max",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3",
]

LAB_OF: dict[str, str] = {
    "gpt-5.5": "OpenAI",
    "gpt-5.6-luna": "OpenAI",
    "gpt-5.6-sol": "OpenAI",
    "gpt-5.6-terra": "OpenAI",
    "gpt-6-astra": "OpenAI",
    "gemini-3.1-pro-preview": "Google",
    "gemini-3.6-flash": "Google",
    "gemini-3.7-flash": "Google",
    "gemini-3.8-flash": "Google",
    "claude-fable-5": "Anthropic",
    "claude-opus-4-6": "Anthropic",
    "claude-opus-4-8": "Anthropic",
    "claude-opus-5": "Anthropic",
    "claude-sonnet-5": "Anthropic",
    "qwen3.8-max": "Qwen",
    "glm-5.3": "ZAI",
    "glm-5.3-flash": "ZAI",
    "kimi-k3": "Moonshot AI",
}

# 机构顺序固定，画混淆矩阵时保证行序一致
LABS: list[str] = ["OpenAI", "Google", "Anthropic", "Qwen", "ZAI", "Moonshot AI"]

LABEL2IDX: dict[str, int] = {m: i for i, m in enumerate(LABELS)}


def label_index(names) -> list[int]:
    return [LABEL2IDX[m] for m in names]


def load_train() -> pd.DataFrame:
    """训练集：id, prompt_id, text, model"""
    df = pd.read_csv(DATA_RAW / "train.csv")
    unknown = set(df["model"]) - set(LABELS)
    if unknown:
        raise ValueError(f"训练集出现未知标签: {sorted(unknown)}")
    return df


def load_test() -> pd.DataFrame:
    """测试集：id, text（交互运行时只有 3 行示例）"""
    return pd.read_csv(DATA_RAW / "test.csv")


def ensure_outputs() -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    return OUTPUTS
