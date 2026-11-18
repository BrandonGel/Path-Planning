"""
Run LaCAM on the 2d map. LaCAM plans point particles on the grid only.
python scripts/mapf/run_lacam.py
"""
from mapf_demo import run_demo

if __name__ == "__main__":
    run_demo("lacam", [(0.0, 0.0)], spaces=(True,), inflate=None)
