"""Independent read-only progress viewer; Ctrl-C never signals the training process."""
import argparse
import json
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--once',action='store_true');args=p.parse_args()
    previous=None
    try:
        while True:
            path=Path(args.output)/'progress.json'
            value=path.read_text() if path.exists() else '{"state":"waiting"}'
            if value!=previous:
                row=json.loads(value)
                print(json.dumps(row,ensure_ascii=False),flush=True);previous=value
            if args.once or json.loads(value)['state'] in ('complete','failed'):break
            time.sleep(2)
    except KeyboardInterrupt:pass


if __name__=='__main__':main()
