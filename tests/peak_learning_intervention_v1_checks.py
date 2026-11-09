import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from peak_learning_intervention_v1 import main
if __name__=='__main__':sys.argv.append('--fixture');main()
