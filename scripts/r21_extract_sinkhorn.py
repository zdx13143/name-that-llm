"""Extract the CPU-only Sinkhorn sections from the audited notebook transcripts."""
from pathlib import Path

ROOT = Path('D:/kaggle/name_that_llm/outputs/codex_runs/r21-plan/analysis')
DEST = Path(__file__).resolve().parent / 'r21_source'


def main():
    DEST.mkdir(exist_ok=True)
    for name, first, second in [('areeb', (928, 930), (947, 961)),
                                ('r16', (838, 840), (857, 871))]:
        lines = (ROOT / (name + '_cells.txt')).read_text(encoding='utf-8').splitlines(keepends=True)
        text = ''.join(lines[first[0]-1:first[1]] + lines[second[0]-1:second[1]])
        (DEST / (name + '_sinkhorn.py.txt')).write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
