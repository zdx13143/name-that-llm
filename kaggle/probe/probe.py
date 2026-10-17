import sys

print("python", sys.version.split()[0])

mods = [
    "torch", "transformers", "tokenizers", "safetensors", "huggingface_hub",
    "sentencepiece", "sklearn", "scipy", "numpy", "pandas", "pyarrow",
]
for m in mods:
    try:
        mod = __import__(m)
        print(f"{m:<18} {getattr(mod, '__version__', '?')}")
    except Exception as exc:
        print(f"{m:<18} MISSING ({type(exc).__name__})")

try:
    import torch
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception:
    pass
