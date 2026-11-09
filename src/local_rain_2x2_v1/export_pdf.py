"""Combine existing PNG figures for review; no analysis or rendering changes."""
import argparse
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--figures', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    names = ('01_training_selection.png', '02_target_peaks.png',
             '03_paired_and_cost.png', '04_reliability.png')
    pages = []
    for name in names:
        with Image.open(args.figures / name) as source:
            pages.append(source.convert('RGB'))
    with args.output.open('xb') as destination:
        pages[0].save(destination, format='PDF', save_all=True,
                      append_images=pages[1:], resolution=150.)
    for page in pages:
        page.close()
    print(f'Saved {len(names)} existing figures to {args.output}', flush=True)


if __name__ == '__main__':
    main()
