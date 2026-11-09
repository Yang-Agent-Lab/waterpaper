"""Keep the retained pilot output inside this checkout's designated run."""
from pathlib import Path

RUN = 'extreme_process_transfer_pilot_20261101_v1'
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def checked_output(path):
    project = PROJECT_ROOT.resolve()
    root = (project/'results'/'experiments'/RUN).resolve()
    output = Path(path).resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError('output escapes approved run root')
    return output, root, 'configured-compute'
