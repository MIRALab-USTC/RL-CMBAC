import sys, os
sys.path.insert(0,os.getcwd())
from mbrl.utils.launch_utils import parse_cmd, run_experiments
if __name__ == '__main__':
    run_experiments(*parse_cmd())