"""The versioned replay --fixture path runs exact labels/peak eligibility checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prob_flood_replay_v1 import main
if __name__=='__main__':
    sys.argv.append('--fixture');main()
