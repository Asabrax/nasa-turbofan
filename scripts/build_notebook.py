"""Build the main presentation; the original builder is build_baseline_notebook.py."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('build_presentation.py')),run_name='__main__')
