"""Render this branch's short learning note with stdlib; no experiment inputs."""
from pathlib import Path
import hashlib
import html
import re

ROOT = Path(__file__).resolve().parents[1]


def inline(text):
    text = html.escape(text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    return re.sub(r'\[([^\]]+)\]\(([^\s)]+)\)', r'<a href="\2">\1</a>', text)


def main():
    source = (ROOT / 'PROJECT_LEARNING.md').read_bytes()
    blocks = []
    for paragraph in source.decode('utf-8').strip().split('\n\n'):
        if paragraph.startswith('# '):
            blocks.append('<h1>' + inline(paragraph[2:]) + '</h1>')
        elif paragraph.startswith('## '):
            blocks.append('<h2>' + inline(paragraph[3:]) + '</h2>')
        else:
            blocks.append('<p>' + inline(paragraph) + '</p>')
    digest = hashlib.sha256(source).hexdigest()
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Name That LLM · R15 学习记录</title><style>
body{max-width:860px;margin:48px auto;padding:0 24px;font:17px/1.85 system-ui,sans-serif;color:#243047;background:#f7f8fa}
h1{font-size:30px;line-height:1.4}h2{margin-top:36px;font-size:23px}a{color:#1757a6}code{background:#e9edf3;padding:2px 5px;border-radius:4px;font-size:.88em}footer{margin-top:40px;font-size:12px;color:#626e82;overflow-wrap:anywhere}
</style><main>''' + '\n'.join(blocks) + f'</main><footer>Markdown SHA256: {digest}</footer></html>\n'
    (ROOT / 'PROJECT_LEARNING.html').write_text(page, encoding='utf-8', newline='\n')
    print('Rendered PROJECT_LEARNING.html; source SHA256 ' + digest)


if __name__ == '__main__':
    main()
