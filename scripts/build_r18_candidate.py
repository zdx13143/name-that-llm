"""Build R18 from the frozen, scored R16 notebook; never executes models."""
import ast
import hashlib
import json
from pathlib import Path
from build_r17_candidate import frozen_bytes, source, BASE, BASE_SHA256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'notebooks/team_e659_r18_adapter_isolation.ipynb'


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('R18 unexpected frozen source: ' + old[:100])
    return text.replace(old, new, 1)


def function_source(script):
    tree = ast.parse(script)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'l24_predict')
    lines = script.splitlines(keepends=True)
    return ''.join(lines[node.decorator_list[0].lineno - 1:node.end_lineno])


def isolate_function(old):
    code = once(old, 'truncated base loaded ONCE;', 'fresh truncated base for EVERY adapter;')
    code = once(code, 'heads, then unmerge.', 'heads, then release the entire base.')
    load = "    m = AutoModel.from_pretrained(spec['path'], config=cfg, dtype=torch.float16, device_map='auto').eval()\n    if spec.get('no_final_norm'): m.norm = torch.nn.Identity()\n    mods = dict(m.named_modules()); first = next(m.parameters()).device; L = spec['layers']\n"
    code = once(code, load, "    L = spec['layers']\n    from r18_contract import Lifecycle\n    lifecycle = Lifecycle(spec['name'])\n")
    code = once(code, '    for ai, ad in enumerate(adapters):\n',
                "    for ai, ad in enumerate(adapters):\n        lifecycle.before_load(ad)\n" +
                ''.join('    ' + line for line in load.replace("; L = spec['layers']", '').splitlines(keepends=True)) +
                '        lifecycle.loaded(m)\n')
    code = once(code, "; deltas = []", '')
    code = once(code, '            deltas.append((W, B, v))   # keep only the low-rank factors; the dense delta is recomputed to unmerge\n',
                '            del W, B, v\n')
    code = once(code, "        if os.path.exists(os.path.join(ad, 'heads.pt')):", "        del sd\n        if os.path.exists(os.path.join(ad, 'heads.pt')):")
    code = once(code, "        for W, B, A_ in deltas:   # unmerge before the next adapter\n            W -= ((B.to(W.device, torch.float32) @ A_.to(W.device, torch.float32)) * scale).to(W.dtype)\n        torch.cuda.empty_cache()\n        del deltas, sd; gc.collect()\n", '')
    for prefix in ('Z_[b]', 'Z[b]'):
        line = next(line for line in code.splitlines(keepends=True) if line.lstrip().startswith(prefix + ' = torch.cat'))
        code = once(code, line, line + '            del hs\n')
    code = once(code, '        log(f\'  l24', '        del m, mods\n        gc.collect(); torch.cuda.empty_cache()\n        lifecycle.released()\n        log(f\'  l24')
    code = once(code, '    del m, mods; gc.collect(); torch.cuda.empty_cache()\n', '')
    compile(code, '<r18-l24>', 'exec')
    return code


def build():
    nb = json.loads(frozen_bytes(BASE, BASE_SHA256))
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    before = [source(c) for c in cells]
    helper = (ROOT / 'scripts/r18_runtime_contract.py').read_text(encoding='utf-8')
    preflight = (ROOT / 'scripts/r18_preflight.py').read_text(encoding='utf-8')
    for filename, text in reversed([('r18_contract.py', helper), ('r18_preflight.py', preflight)]):
        nb['cells'].insert(nb['cells'].index(cells[1]), {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': ('%%writefile ' + filename + '\n' + text).splitlines(keepends=True)})
    cells[0]['source'] = ('import time as _r18time\nR18_START = _r18time.monotonic()\n' + before[0]).splitlines(keepends=True)
    cells[1]['source'] = (before[1] + '\nfrom r18_contract import run_logged, remaining_seconds, validate_comp_log, validate_main_log\n').splitlines(keepends=True)
    infer = before[3].split('\n', 1)[1]
    original_function = function_source(infer)
    replacement = isolate_function(original_function)
    cells[3]['source'] = ('%%writefile infer.py\n' + once(infer, original_function, replacement)).splitlines(keepends=True)
    cells[5]['source'] = once(before[5], "    subprocess.run([sys.executable, 'comp_stack.py'], check=True)\n",
        "    _r18_comp_log = run_logged([sys.executable, 'comp_stack.py'], '/kaggle/working/r18_comp.log', remaining_seconds(R18_START))\n    R18_COMP = validate_comp_log(_r18_comp_log)\n").splitlines(keepends=True)
    # Lift the exact model list out of the rerun guard so Save can use identical specs.
    main = before[6]
    end = main.index('    import pandas as pd\n')
    models = main[len('if RERUN:\n'):end]
    models = ''.join(line[4:] if line.startswith('    ') else line for line in models.splitlines(keepends=True))
    main = models + 'if RERUN:\n' + main[end:]
    main = once(main, '    ATLAS_BUDGET = int(8.3 * 3600 - COMP_SEC - 600)\n', '    ATLAS_BUDGET = int(remaining_seconds(R18_START) - 600)\n')
    main = once(main, '    subprocess.run(cmd, check=True)\n',
        "    _r18_main_log = run_logged(cmd, '/kaggle/working/r18_main.log', remaining_seconds(R18_START))\n    R18_MAIN = validate_main_log(_r18_main_log)\n    with open('/kaggle/working/r18_runtime_receipt.json', 'w') as _f:\n        json.dump({'comp': R18_COMP, 'main': R18_MAIN, 'wall_seconds': _r18time.monotonic() - R18_START}, _f, indent=2)\n")
    main = once(main, 'else:\n    _r14placeholder',
        "else:\n    with open('/kaggle/working/r18_preflight_specs.json', 'w') as _f:\n        json.dump({'specs': MODELS[:2], 'labels': _R14_LABELS}, _f)\n    run_logged([sys.executable, 'r18_preflight.py'], '/kaggle/working/r18_preflight.log', remaining_seconds(R18_START, 900))\n    _r14placeholder")
    cells[6]['source'] = main.splitlines(keepends=True)
    if source(cells[2]) != before[2] or source(cells[4]) != before[4]:
        raise ValueError('R18 modified Gaussian or COMP body')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            text = source(cell)
            compile(text.split('\n', 1)[1] if text.startswith('%%writefile ') else text, '<r18-cell>', 'exec')
    nb['cells'][0]['source'] = ('''# R18: isolated base weights for each LoRA adapter

Baseline: R16 V2 / 363487968, Public Score 0.659. R18 has no score yet.
Each main LoRA adapter now starts from a fresh pretrained base. The original
merge, pooling, head averaging and within-adapter overflow retry are retained.
All R16 prediction settings, typography strength 0.5 and 30 inputs remain fixed.
The known missing main.ll8v5 and comp.lora24_public remain absent.
Normal Save runs a bounded two-artificial-passage preflight for all 12 Qwen v5
adapters, then writes three placeholder rows. This is not an accuracy test.
The formal rerun enforces a total 8-hour Python ceiling and records actual ATLAS
coverage; extra reload time may reduce coverage. No complete isolated OOF exists.
One formal submission only after all release gates pass. Original attribution follows.
''').splitlines(keepends=True)
    nb['metadata'].pop('r16', None)
    nb['metadata']['r18'] = {'scored_base_sha256': BASE_SHA256, 'scored_base_version': 363487968,
        'scored_base_public_score': 0.659, 'typography_strength': 0.5,
        'change': 'fresh pretrained base per main l24 adapter',
        'omitted_members': ['comp.lora24_public', 'main.ll8v5'], 'full_frozen_ensemble': False,
        'local_complete_ensemble_cv': 'not_evaluated', 'formal_submissions_budget': 1,
        'saves_budget': 2, 'public_score': None}
    return nb


if __name__ == '__main__':
    nb = build()
    packed = (json.dumps(nb, ensure_ascii=False, indent=1) + '\n').encode('utf-8')
    OUTPUT.write_bytes(packed)
    recipe = {'candidate': OUTPUT.relative_to(ROOT).as_posix(), 'candidate_sha256': hashlib.sha256(packed).hexdigest(),
        'base_sha256': BASE_SHA256, 'code_sha256': [hashlib.sha256(source(c).encode()).hexdigest() for c in nb['cells'] if c['cell_type'] == 'code'],
        'inputs': nb['metadata']['kaggle']['dataSources'], 'release_gate': 'local_checks_then_real_asset_save_preflight',
        'formal_submissions': 0, 'saves': 0}
    (ROOT / 'configs/r18_recipe_20261101.json').write_text(json.dumps(recipe, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'candidate_sha256': recipe['candidate_sha256'], 'code_cells': len(recipe['code_sha256']), 'inputs': len(recipe['inputs'])}))
