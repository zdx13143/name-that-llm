"""Freeze exact CPU blocks from the two audited notebook sources."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    'r16': ROOT / 'notebooks/team_e654_r16_typography_half_runnable.ipynb',
    'areeb': Path('D:/kaggle/name_that_llm/outputs/codex_runs/r21-plan/sources/areebbb__ntl-we-back-0-666/ntl-we-back-0-666.ipynb'),
}
ANCHORS = {
    'extra': ('EXTRA_DF = pd.DataFrame', '# ---------------- 2. TF-IDF centred LR'),
    'tfidf': ('if a.tfidf_w > 0:', 'if a.svc_w > 0:'),
    'svc4': ('if a.svc_w > 0:', 'if a.distort_w > 0:'),
    'distort': ('if a.distort_w > 0:', '# ---------------- 3. LM embeddings'),
}


def extract(path):
    raw = path.read_bytes()
    notebook = json.loads(raw)
    source = next(''.join(c.get('source', [])) for c in notebook['cells']
                  if 'if a.svc_w > 0:' in ''.join(c.get('source', [])))
    blocks = {}
    for name, (start, end) in ANCHORS.items():
        if source.count(start) != 1 or source.count(end) != 1:
            raise ValueError('source anchors changed: ' + name)
        blocks[name] = source[source.index(start):source.index(end)]
    blend = 'tot = sum(LP.values())\n'
    if source.count(blend) != 1:
        raise ValueError('blend anchor changed')
    blocks['blend'] = blend
    return hashlib.sha256(raw).hexdigest(), blocks


def main():
    target = ROOT / 'scripts/r21_source'
    target.mkdir(exist_ok=True)
    for version, path in SOURCES.items():
        sha, blocks = extract(path)
        for name, code in blocks.items():
            (target / f'{version}_{name}.py.txt').write_text(code, encoding='utf-8', newline='\n')
        print(version, sha)


if __name__ == '__main__':
    main()
