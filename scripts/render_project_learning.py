"""Render the root learning journal as a self-contained, offline HTML page.

Standard-library only. Reads the Markdown journal, never experiment data.
This deliberately supports only the Markdown constructs used in the journal.
"""
from pathlib import Path
import hashlib
import html
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "PROJECT_LEARNING.md"
DESTINATION = ROOT / "PROJECT_LEARNING.html"


def inline(text):
    tokens = []

    def save(value):
        tokens.append(value)
        return f"\x00{len(tokens) - 1}\x00"

    text = re.sub(r"`([^`]+)`", lambda m: save("<code>" + html.escape(m[1]) + "</code>"), text)
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^\s)]+)\)",
                  lambda m: '<a href="' + m[2] + '">' + m[1] + "</a>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: tokens[int(m[1])], text)


def score_chart(markdown):
    match = re.search(r"(?m)^\| pooled BA \|(.+)\|\s*$", markdown)
    if not match:
        raise ValueError("The score chart requires the journal's pooled BA comparison row")
    values = [float(cell.strip()) for cell in match[1].split("|")]
    preceding = markdown[:match.start()].rstrip().splitlines()
    while preceding and preceding[-1].startswith("|"):
        header = preceding.pop()
    labels = [cell.strip() for cell in header.strip("|").split("|")][1:]
    if len(labels) != len(values) or any(not 0 <= value <= 1 for value in values):
        raise ValueError("Score labels and finite BA values must match")
    threshold_match = re.search(r"预登记要求 pooled BA 至少 ([0-9.]+)", markdown)
    if not threshold_match:
        raise ValueError("Missing preregistered pooled BA threshold")
    threshold = float(threshold_match[1])
    rows = [(html.escape(label), value, "new" if i == len(values) - 1 else "old")
            for i, (label, value) in enumerate(zip(labels, values))]
    rows.append(("预登记 pooled BA 门槛", threshold, "gate"))
    scale = max(0.4, threshold, *values)
    bars = []
    for label, value, kind in rows:
        bars.append(f'<div class="bar-row {kind}"><div class="bar-label">{label}'
                    f'<strong>{value:.6f}</strong></div><div class="bar-track">'
                    f'<span style="width:{value / scale * 100:.4f}%"></span></div></div>')
    return '<figure class="chart"><figcaption>同一开发集 · pooled balanced accuracy</figcaption>' + "".join(bars) + f'<div class="axis"><span>0</span><span>{scale:g}</span></div><p class="chart-note">横轴从 0 起。图中仅展示 pooled BA；是否提交还取决于各折表现、配对比较和完整性检查。这张图不是 Kaggle 榜单。</p></figure>'


FLOW = '<div class="flow" aria-label="实验流程">' + "".join(
    f'<div><span>{i:02d}</span><b>{label}</b></div>' for i, label in enumerate(
        ["提出假设", "固定验证", "生成特征", "折内拟合", "OOF 评分", "门槛决策"], 1)) + '</div>'

CALCULATOR = '''<section class="calculator" aria-labelledby="calculator-title">
<div class="eyebrow">动手理解 · 虚构的 3 类示例</div><h3 id="calculator-title">同样一份预测，为什么会有两个分数？</h3>
<p>真实数量固定为 A=100、B=10、C=10。拖动答对数量，观察两种指标的不同权重。</p>
<div class="calc-grid">
<label for="correct-a">A 类答对 <output id="value-a" for="correct-a">90 / 100</output><input id="correct-a" type="range" min="0" max="100" value="90"></label>
<label for="correct-b">B 类答对 <output id="value-b" for="correct-b">4 / 10</output><input id="correct-b" type="range" min="0" max="10" value="4"></label>
<label for="correct-c">C 类答对 <output id="value-c" for="correct-c">2 / 10</output><input id="correct-c" type="range" min="0" max="10" value="2"></label>
</div><div class="calc-results" aria-live="polite"><div><small>普通 accuracy</small><strong id="accuracy">80.0%</strong><span>所有样本一起计数</span></div><div><small>Balanced accuracy</small><strong id="balanced">50.0%</strong><span>每一类获得同等权重</span></div></div>
<p id="calc-formula" class="formula">accuracy = 96 / 120；BA = (90% + 40% + 20%) / 3</p>
<button type="button" id="reset-example">恢复初始例子</button><noscript><p>启用 JavaScript 可操作滑块；初始例子的静态计算已展示在上方。</p></noscript></section>'''


def render(markdown):
    lines = markdown.splitlines()
    result, toc = [], []
    i, section = 0, 0
    markers = {"<!-- score-chart -->": score_chart(markdown), "<!-- learning-flow -->": FLOW,
               "<!-- learning-calculator -->": CALCULATOR}
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.startswith("# "):
            i += 1
        elif line in markers:
            result.append(markers[line]); i += 1
        elif line.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i]); i += 1
            if i == len(lines):
                raise ValueError("Unclosed code block")
            result.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>"); i += 1
        elif line.startswith("## "):
            section += 1
            title = line[3:]
            toc.append(f'<a href="#section-{section}">{html.escape(title)}</a>')
            result.append(f'<h2 id="section-{section}">{inline(title)}</h2>'); i += 1
        elif line.startswith("### "):
            result.append("<h3>" + inline(line[4:]) + "</h3>"); i += 1
        elif line.startswith("| "):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            table = '<div class="table-wrap" tabindex="0" role="region" aria-label="可横向滚动的数据表"><table><thead><tr>'
            table += "".join("<th scope=\"col\">" + inline(c) + "</th>" for c in rows[0])
            table += "</tr></thead><tbody>"
            table += "".join("<tr>" + "".join("<td>" + inline(c) + "</td>" for c in r) + "</tr>" for r in rows[1:])
            result.append(table + "</tbody></table></div>")
        elif line.startswith("> "):
            result.append('<aside class="status">' + inline(line[2:]) + '</aside>'); i += 1
        elif line.startswith("- ") or re.match(r"^\d+\. ", line):
            ordered = not line.startswith("- ")
            tag = "ol" if ordered else "ul"
            items = []
            pattern = r"^\d+\. " if ordered else r"^- "
            while i < len(lines) and re.match(pattern, lines[i]):
                items.append("<li>" + inline(re.sub(pattern, "", lines[i])) + "</li>"); i += 1
            result.append(f"<{tag}>" + "".join(items) + f"</{tag}>")
        else:
            paragraph = [line]; i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||>|-|\d+\.|```|<!--)", lines[i]):
                paragraph.append(lines[i]); i += 1
            result.append("<p>" + inline(" ".join(paragraph)) + "</p>")
    return "\n".join(result), "\n".join(toc)


STYLE = '''
:root{color-scheme:light;--ink:#173331;--muted:#647572;--paper:#f6f5ef;--line:#dce4dc;--teal:#176b60;--amber:#a56c26;--white:#fffefa}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:25px}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Segoe UI","Microsoft YaHei",sans-serif;font-size:16px;line-height:1.9}a{color:var(--teal);text-underline-offset:4px}a:hover{color:#114d45}a:focus-visible,button:focus-visible,input:focus-visible,.table-wrap:focus-visible{outline:3px solid #d3943c;outline-offset:4px}.skip{position:fixed;top:-100px;left:10px;z-index:5;background:white;padding:10px}.skip:focus{top:10px}.hero{padding:68px max(28px,calc((100vw - 1210px)/2)) 55px;background:var(--ink);color:#f3f4e9;overflow:hidden}.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;font-weight:700}.hero .eyebrow{color:#b2d4c7}.hero h1{font-size:clamp(30px,4.1vw,54px);line-height:1.3;letter-spacing:-1.5px;margin:16px 0}.hero p{color:#c4d2cc;max-width:670px;margin:0}.hero .tag{display:inline-block;border:1px solid #67817b;border-radius:30px;padding:3px 13px;font-size:13px;margin-top:23px}.layout{max-width:1260px;margin:auto;display:grid;grid-template-columns:250px minmax(0,1fr);gap:52px;padding:42px 25px 80px}.sidebar{position:sticky;top:25px;align-self:start}.sidebar>span{color:var(--muted);font-size:12px;letter-spacing:2px}.sidebar nav{margin-top:18px;display:grid;gap:5px}.sidebar a{display:block;text-decoration:none;font-size:13px;line-height:1.7;padding:8px 12px;border-left:2px solid var(--line)}.sidebar a:hover,.sidebar a.active{background:#e7eee5;border-color:var(--teal)}.sidebar .source{margin-top:25px;border-left:0}main{min-width:0}h2{font-size:27px;line-height:1.5;margin:68px 0 24px;letter-spacing:-.6px;padding-top:15px;border-top:1px solid var(--line)}h3{font-size:19px;line-height:1.7;margin:30px 0 12px}p{margin:16px 0}strong{font-weight:650}li{padding:3px 0}li+li{margin-top:7px}.status{border-left:4px solid var(--teal);background:#e8eee5;padding:19px 23px;font-size:14px;color:#36534b;border-radius:0 8px 8px 0}.table-wrap{overflow:auto;margin:24px 0;border:1px solid var(--line);border-radius:10px;background:var(--white)}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.8;min-width:490px}th,td{text-align:left;padding:13px 15px;vertical-align:top;border-bottom:1px solid #e8ece4}th{background:#eaf0e8;font-weight:650}tr:last-child td{border-bottom:0}code{font-family:Consolas,monospace;font-size:.88em;background:#e9ede4;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}pre{background:#153c36;color:#e5eee1;padding:22px;border-radius:10px;overflow:auto;line-height:1.7;font-size:14px}pre code{background:none;padding:0;white-space:pre;overflow-wrap:normal}.chart{margin:28px 0;padding:27px 30px;background:var(--white);border:1px solid var(--line);border-radius:14px}figcaption{font-size:14px;font-weight:650;margin-bottom:21px}.bar-row{margin:14px 0}.bar-label{display:flex;justify-content:space-between;gap:12px;font-size:13px}.bar-label strong{font-variant-numeric:tabular-nums}.bar-track{background:#ebeee6;border-radius:4px;height:13px;margin-top:7px;overflow:hidden}.bar-track span{display:block;height:100%;background:#899e96;border-radius:4px}.new .bar-track span{background:var(--teal)}.gate .bar-track span{background:repeating-linear-gradient(45deg,#bd904f,#bd904f 5px,#dfc49a 5px,#dfc49a 10px)}.axis{display:flex;justify-content:space-between;font-size:11px;color:var(--muted)}.chart-note{font-size:12px;color:var(--muted);margin-bottom:0}.flow{display:grid;grid-template-columns:repeat(6,1fr);gap:7px;margin:28px 0}.flow div{padding:15px 5px 15px 10px;border-top:2px solid #81a596;background:#eaf0e6}.flow span{font:12px Consolas,monospace;color:var(--teal);display:block}.flow b{font-size:12px;font-weight:600}.calculator{background:#fff8e9;border:1px solid #e7d9ba;border-radius:14px;padding:28px;margin:28px 0}.calculator .eyebrow{color:var(--amber)}.calculator h3{margin:10px 0}.calculator p{font-size:14px}.calc-grid{display:grid;gap:17px;margin:22px 0}.calc-grid label{font-size:14px}.calc-grid output{float:right;font-variant-numeric:tabular-nums}input[type=range]{width:100%;accent-color:var(--teal);display:block;margin:12px 0 0;min-height:24px}.calc-results{display:grid;grid-template-columns:1fr 1fr;gap:15px}.calc-results div{background:var(--white);border-radius:10px;padding:17px 20px}.calc-results small,.calc-results span{display:block;font-size:12px;color:var(--muted)}.calc-results strong{display:block;font-size:36px;color:var(--teal);line-height:1.5;font-variant-numeric:tabular-nums}.formula{font-variant-numeric:tabular-nums;overflow-wrap:anywhere}button{cursor:pointer;border:1px solid #9aab9f;border-radius:7px;background:transparent;padding:10px 16px;color:var(--ink);font:inherit;font-size:13px}button:hover{background:#e9eee1}footer{border-top:1px solid var(--line);padding:25px;text-align:center;font-size:12px;color:var(--muted);overflow-wrap:anywhere}.page-top{display:inline-block;margin-top:35px;font-size:13px}
@media(max-width:900px){.layout{grid-template-columns:1fr;gap:25px;max-width:810px}.sidebar{position:static}.sidebar nav{grid-template-columns:1fr 1fr;gap:3px}.sidebar .source{margin-top:12px}.hero{padding:40px 25px}.sidebar>span{display:block}h2{margin-top:45px}}
@media(max-width:540px){body{font-size:15px}.layout{padding:25px 18px 45px}.hero{padding:35px 20px}.hero h1{font-size:32px}.sidebar nav{grid-template-columns:1fr}.sidebar a{padding:6px 10px}.flow{grid-template-columns:repeat(3,1fr)}.chart,.calculator{padding:20px 17px}.bar-label{font-size:12px}.calc-results{gap:8px}.calc-results div{padding:13px 10px}.calc-results strong{font-size:31px}h2{font-size:23px}.status{padding:16px}table{min-width:480px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{.sidebar,.skip,button,.page-top{display:none}.layout{display:block;padding:0}.hero{background:white;color:var(--ink);padding:0}.hero p,.hero .eyebrow{color:var(--ink)}body{background:white;font-size:11pt}.table-wrap{overflow:visible}table{min-width:0;font-size:9pt}.chart,.calculator{break-inside:avoid}h2{break-after:avoid}a{color:inherit}}
'''

SCRIPT = '''
const keys=['a','b','c'], totals=[100,10,10];
function update(){const values=keys.map(k=>Number(document.getElementById('correct-'+k).value));
keys.forEach((k,i)=>document.getElementById('value-'+k).textContent=values[i]+' / '+totals[i]);
const sum=values.reduce((a,b)=>a+b,0), recalls=values.map((n,i)=>100*n/totals[i]);
document.getElementById('accuracy').textContent=(100*sum/120).toFixed(1)+'%';
document.getElementById('balanced').textContent=(recalls.reduce((a,b)=>a+b,0)/3).toFixed(1)+'%';
document.getElementById('calc-formula').textContent='accuracy = '+sum+' / 120；BA = ('+recalls.map(n=>n.toFixed(0)+'%').join(' + ')+') / 3';}
keys.forEach(k=>document.getElementById('correct-'+k).addEventListener('input',update));
document.getElementById('reset-example').addEventListener('click',()=>{[90,4,2].forEach((n,i)=>document.getElementById('correct-'+keys[i]).value=n);update()});
if('IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>{entries.forEach(e=>{if(e.isIntersecting){document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('active',a.hash==='#'+e.target.id))}})},{rootMargin:'-5% 0px -75% 0px'});document.querySelectorAll('main h2').forEach(h=>observer.observe(h))}update();
'''


def main():
    raw = SOURCE.read_bytes()
    markdown = raw.decode("utf-8")
    body, toc = render(markdown)
    fingerprint = hashlib.sha256(raw).hexdigest()
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Name That LLM 实验过程与 AI 入门笔记：用实际结果理解验证、特征、分类与不确定性。">
<meta name="source-sha256" content="{fingerprint}"><title>Name That LLM · 项目进展与 AI 学习笔记</title><style>{STYLE}</style></head>
<body><a class="skip" href="#content">跳到正文</a><header class="hero" id="top"><div class="eyebrow">NAME THAT LLM / 实验与学习</div><h1>读懂每一次实验。</h1><p>从一段文本到一个判断，跟着真实项目理解 AI。记录做过什么，也理解为什么这样做、结果能说明什么。</p><span class="tag">项目进展 × 小白学习笔记 · 离线版</span></header>
<div class="layout"><aside class="sidebar"><span>阅读目录</span><nav aria-label="章节目录">{toc}</nav><a class="source" href="PROJECT_LEARNING.md">查看 Markdown 原文 ↗</a></aside><main id="content">{body}<a class="page-top" href="#top">回到顶部 ↑</a></main></div>
<footer>由 PROJECT_LEARNING.md 生成 · 无外部资源依赖<br>文字源 SHA256：{fingerprint}</footer><script>{SCRIPT}</script></body></html>'''
    with DESTINATION.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(page)
    print(f"Rendered {DESTINATION.name}; source SHA256={fingerprint}; sections={len(re.findall(r'<h2 ', body))}")


if __name__ == "__main__":
    main()
