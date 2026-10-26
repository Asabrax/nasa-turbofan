"""Refresh the result summary and render the executed notebook for inspection."""
import html
import json
import argparse
from pathlib import Path
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('notebook', nargs='?', default='explore.ipynb')
args = parser.parse_args()
path = ROOT / args.notebook
nb = json.loads(path.read_text())
if nb.get('metadata', {}).get('presentation'):
    import runpy
    import sys
    sys.argv = [str(ROOT / 'scripts/render_presentation.py'), args.notebook]
    runpy.run_path(str(ROOT / 'scripts/render_presentation.py'), run_name='__main__')
    raise SystemExit(0)
out = ROOT / nb.get('metadata', {}).get('artifact_dir', 'outputs')
result = json.loads((out / 'results.json').read_text())
if 'selected_config' in result:
    selected_metrics = next(row for row in result['test_metrics']
                            if row['model'] == 'selected' and row['dataset'] == 'ALL' and row['threshold'] == 30)
    selected_name = result['selected_config']['name']
    test_count = result['test_engines']
    test_result = {'RMSE_cycles': selected_metrics['RMSE'], 'MAE_cycles': selected_metrics['MAE']}
    precision = selected_metrics['precision']
    recall = selected_metrics['recall']
    population = 'across FD001–FD004 (true, uncapped labels)'
elif 'selected_family' in result:
    selected_metrics = next(row for row in result['test_metrics']
                            if row['family'] == result['selected_family'] and row['dataset'] == 'ALL')
    selected_name = result['selected_candidate']
    test_count = result['test_engines']
    test_result = selected_metrics
    precision = selected_metrics['precision']
    recall = selected_metrics['recall']
    population = 'across FD001–FD004'
else:
    selected_name = result['selected_model']
    test_count = 100
    test_result = result['test']
    precision = result['maintenance']['precision']
    recall = result['maintenance']['recall']
    population = 'in FD001'
summary = (
    '## Executed results\n\n'
    f"Grouped cross-validation selected **{selected_name}**. "
    f"On {test_count} test-engine endpoints {population}, RMSE is **{test_result['RMSE_cycles']:.2f} cycles** "
    f"and MAE is **{test_result['MAE_cycles']:.2f} cycles**. "
    f"The illustrative 30-cycle maintenance rule has **{precision:.1%} precision** "
    f"and **{recall:.1%} recall**. "
    'These results describe this simulation benchmark; see the evaluation limits below.'
)
if 'selected_config' in result:
    old = next(row for row in result['test_metrics'] if row['model'] == 'Random forest' and row['dataset'] == 'ALL')
    tuned = next(row for row in result['test_metrics'] if row['model'] == 'selected' and row['dataset'] == 'ALL'
                 and row['threshold'] == result['alert_threshold'])
    audit_rows = {(row['model'], row['protocol']): row for row in result['audit_metrics'] if row['threshold'] == 30}
    summary += (
        f"\n\nThe previous best test RMSE was {old['RMSE']:.2f} cycles: this is a "
        f"{(1-test_result['RMSE_cycles']/old['RMSE'])*100:.1f}% reduction. "
        f"On untouched audit engines, primary-window RMSE improved from "
        f"{audit_rows['reference', 'primary']['RMSE']:.2f} to {audit_rows['selected', 'primary']['RMSE']:.2f}. "
        f"**Tradeoff:** broad early-life audit RMSE worsened from {audit_rows['reference', 'broad']['RMSE']:.2f} "
        f"to {audit_rows['selected', 'broad']['RMSE']:.2f} cycles. This is not a uniform improvement at every engine age."
        f"\n\nThe validation-selected alert threshold is **{result['alert_threshold']:.2f} predicted cycles** "
        f"for a true 30-cycle event. On the official test it yields **{tuned['recall']:.1%} recall** "
        f"and **{tuned['precision']:.1%} precision**, with {tuned['FN']} missed near-failure engines and "
        f"{tuned['FP']} false alerts. Test targets remain uncapped throughout."
    )
nb['cells'] = [cell for cell in nb['cells'] if cell.get('id') != 'executed-summary']
nb['cells'].insert(1, {'cell_type': 'markdown', 'id': 'executed-summary', 'metadata': {}, 'source': summary.splitlines(True)})
path.write_text(json.dumps(nb, indent=1) + '\n')
renderer = MarkdownIt()
parts = ['<!doctype html><html><head><meta charset="utf-8"><title>NASA RUL notebook preview</title>',
         '<style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#172033}pre{overflow:auto;background:#f4f6f8;padding:16px;font-size:13px}img{max-width:100%}table{border-collapse:collapse;font-size:14px}td,th{padding:6px 12px;border-bottom:1px solid #ddd}</style></head><body>']
code_count = 0
for cell in nb['cells']:
    source = ''.join(cell['source'])
    if cell['cell_type'] == 'markdown':
        parts.append(renderer.render(source))
    else:
        code_count += 1
        assert cell['execution_count'] == code_count
        parts.append('<pre>' + html.escape(source) + '</pre>')
        for output in cell['outputs']:
            assert output['output_type'] != 'error'
            if output['output_type'] == 'stream':
                parts.append('<pre>' + html.escape(''.join(output['text'])) + '</pre>')
            else:
                data = output['data']
                if 'image/png' in data:
                    parts.append('<img alt="Notebook chart" src="data:image/png;base64,' + data['image/png'] + '">')
                elif 'text/html' in data:
                    parts.append(''.join(data['text/html']))
                elif 'text/plain' in data:
                    parts.append('<pre>' + html.escape(''.join(data['text/plain'])) + '</pre>')
parts.append('</body></html>')
(out / 'notebook_preview.html').write_text('\n'.join(parts))
print(f'Checked {code_count} executed code cells; wrote HTML preview and refreshed summary.')
