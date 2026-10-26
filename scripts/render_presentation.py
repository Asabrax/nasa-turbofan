"""Render the executed presentation notebook as a self-contained local HTML page."""
from pathlib import Path
import json,html,re,argparse
from markdown_it import MarkdownIt
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('notebook',nargs='?',default='explore.ipynb')
args=parser.parse_args()
nb=json.loads((ROOT/args.notebook).read_text())
md=MarkdownIt()
sections=[]
parts=[]
code_count=0
image_count=0
for cell in nb['cells']:
    text=''.join(cell['source'])
    if cell['cell_type']=='markdown':
        markup=md.render(text)
        def heading(match):
            label=match.group(1)
            slug=re.sub(r'[^a-z0-9]+','-',re.sub('<[^>]+>','',label).lower()).strip('-')
            sections.append((slug,label))
            return f'<h2 id="{slug}">{label}</h2>'
        markup=re.sub(r'<h2>(.*?)</h2>',heading,markup)
        parts.append('<div class="narrative">'+markup+'</div>')
    else:
        code_count+=1
        assert cell['execution_count'] is not None
        parts.append('<details class="code"><summary>View reproducible code</summary><pre><code>'+html.escape(text)+'</code></pre></details>')
        for output in cell['outputs']:
            assert output['output_type']!='error'
            if output['output_type']=='stream':
                parts.append('<pre class="status">'+html.escape(''.join(output['text']))+'</pre>')
            else:
                data=output['data']
                if 'image/png' in data:
                    image_count+=1
                    parts.append('<figure><img alt="Presentation chart '+str(image_count)+'" src="data:image/png;base64,'+data['image/png']+'"></figure>')
                elif 'text/html' in data:
                    parts.append('<div class="table-wrap">'+''.join(data['text/html'])+'</div>')
                elif 'text/plain' in data:
                    parts.append('<pre class="status">'+html.escape(''.join(data['text/plain']))+'</pre>')
nav=''.join(f'<a href="#{slug}">{label}</a>' for slug,label in sections)
css='''
:root{color-scheme:light;--ink:#172b4d;--muted:#52657b;--accent:#087f8c;--line:#dfe6ed}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f4f7fa;color:var(--ink);font:17px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}
.sidebar{position:fixed;inset:0 auto 0 0;width:260px;background:#fff;border-right:1px solid var(--line);padding:28px 20px;overflow:auto}
.brand{font-weight:750;letter-spacing:.08em;font-size:12px;text-transform:uppercase;color:var(--accent);margin-bottom:20px}
nav a{display:block;text-decoration:none;color:var(--muted);font-size:13px;padding:8px 0;line-height:1.45}nav a:hover{color:var(--accent)}
main{max-width:1160px;margin:0 auto 0 280px;padding:48px 46px 80px;background:white;min-height:100vh}
h1{font-size:46px;line-height:1.14;letter-spacing:-.035em;max-width:860px;margin:12px 0 20px}h2{font-size:29px;line-height:1.3;margin:64px 0 22px;padding-top:24px;border-top:2px solid var(--line);scroll-margin-top:20px}
h3{font-size:18px;color:var(--muted);font-weight:500;margin:0 0 25px}p{max-width:900px;margin:18px 0}a{color:var(--accent)}li{margin:9px 0}strong{font-weight:700}
figure{margin:26px 0;padding:12px 0}img{display:block;width:100%;height:auto}.table-wrap{overflow-x:auto;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:15px;text-align:right}thead{background:#edf4f6}th,td{padding:11px 14px;border:0;border-bottom:1px solid var(--line)}thead th{text-align:right}tbody th{text-align:left;font-weight:500}tr:hover{background:#f6f9fb}
.code{margin:12px 0;color:var(--muted);border:1px solid var(--line);border-radius:6px;font-size:13px}.code summary{cursor:pointer;padding:10px 14px}.code pre{margin:0;border-top:1px solid var(--line)}pre{white-space:pre;overflow:auto;padding:16px;background:#f5f7fa;font-size:12px;line-height:1.5}.status{white-space:pre-wrap;color:var(--muted);font-size:13px}code{font-size:.87em}footer{font-size:13px;color:var(--muted);border-top:1px solid var(--line);margin-top:45px;padding-top:18px}
@media(min-width:1550px){main{margin-left:calc(260px + (100vw - 1420px)/2)}}
@media(max-width:1000px){.sidebar{position:static;width:auto;padding:18px 24px;border-right:0;border-bottom:1px solid var(--line)}nav{display:none}.brand{margin:0}main{margin:0;padding:28px 24px}h1{font-size:36px}h2{font-size:25px}body{font-size:16px}}
@media print{body{background:white;font-size:11pt}.sidebar,.code{display:none}main{margin:0;padding:0;max-width:none}h1{font-size:28pt}h2{font-size:20pt;break-before:page;margin-top:0}figure,table{break-inside:avoid}a{color:inherit;text-decoration:none}}
'''
page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NASA predictive maintenance · Presentation</title><style>'+css+'</style></head><body><aside class="sidebar"><div class="brand">Predictive maintenance<br>NASA C-MAPSS</div><nav aria-label="Presentation sections">'+nav+'</nav></aside><main>'+''.join(parts)+'<footer>Presentation source: <a href="explore.ipynb">explore.ipynb</a> (executed). Original baseline and EDA: <a href="baseline.ipynb">baseline.ipynb</a>. Local evidence, source code and supporting notebooks remain linked in the appendix.</footer></main></body></html>'
path=ROOT/'presentation.html'
path.write_text(page)
print(f'Wrote {path}: {code_count} executed cells, {image_count} embedded charts, {len(sections)} navigation links.')
