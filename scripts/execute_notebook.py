"""Execute this notebook with installed IPython and retain outputs without nbclient.

This is a sequential runner for this Python-only notebook, not a general Jupyter
kernel replacement. It fails immediately on cell errors and saves PNG figures.
"""
import base64
import io
import json
import os
import argparse
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display, Image

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser()
parser.add_argument('notebook', nargs='?', default='explore.ipynb')
args = parser.parse_args()
path = ROOT / args.notebook
notebook = json.loads(path.read_text())
artifact_dir = ROOT / notebook.get('metadata', {}).get('artifact_dir', 'outputs')
artifact_dir.mkdir(parents=True, exist_ok=True)
shell = InteractiveShell.instance()
figure_count = 0

def show(*args, **kwargs):
    global figure_count
    for number in plt.get_fignums():
        figure_count += 1
        fig = plt.figure(number)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
        (artifact_dir / f'figure_{figure_count:02d}.png').write_bytes(buffer.getvalue())
        display(Image(data=buffer.getvalue()))
    plt.close('all')

plt.show = show

class OrderedStream:
    """Record stream writes in their actual order relative to display events."""
    encoding = 'utf-8'

    def __init__(self, outputs, name):
        self.outputs, self.name = outputs, name

    def write(self, value):
        if value:
            if self.outputs and self.outputs[-1].get('name') == self.name:
                self.outputs[-1]['text'].append(value)
            else:
                self.outputs.append({'output_type': 'stream', 'name': self.name, 'text': [value]})
        return len(value)

    def flush(self):
        pass

    def isatty(self):
        return False


def execute_cell(source):
    outputs = []
    original_publish = shell.display_pub.publish
    def publish(data, metadata=None, **kwargs):
        data = dict(data)
        for mime, value in data.items():
            if isinstance(value, bytes):
                data[mime] = base64.b64encode(value).decode()
        outputs.append({'output_type': 'display_data', 'data': data, 'metadata': metadata or {}})
    shell.display_pub.publish = publish
    try:
        with redirect_stdout(OrderedStream(outputs, 'stdout')), redirect_stderr(OrderedStream(outputs, 'stderr')):
            result = shell.run_cell(source, store_history=False)
    finally:
        shell.display_pub.publish = original_publish
    return result, outputs

count = 0
for cell in notebook['cells']:
    assert cell['cell_type'] in ('markdown', 'code') and isinstance(cell['source'], list)
    if cell['cell_type'] != 'code':
        continue
    count += 1
    print(f'Executing cell {count}', flush=True)
    result, outputs = execute_cell(''.join(cell['source']))
    if result.error_before_exec or result.error_in_exec:
        for output in outputs:
            if output['output_type'] == 'stream':
                print(''.join(output['text']))
        raise RuntimeError(f'Cell {count} failed') from (result.error_before_exec or result.error_in_exec)
    cell['execution_count'] = count
    cell['outputs'] = outputs
path.write_text(json.dumps(notebook, indent=1) + '\n')
print(f'Saved {count} executed cells and {figure_count} figures to {path}')
