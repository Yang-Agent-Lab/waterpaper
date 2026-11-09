import argparse
import json
from pathlib import Path
from .plot import render_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--include-diagnostics',action='store_true',help='Explicit reproduction of the historical fourth page; default omits it')
    args = parser.parse_args()
    render_report(json.loads(Path(args.bundle).read_text()), args.out,include_diagnostics=args.include_diagnostics)


if __name__ == '__main__': main()
