"""
Run LaCAM (random variant) on the 2d map, obstacles inflated by 1. It plans point particles on the grid
only and runs until the time limit (no iteration cap).
python scripts/mapf/run_lacam_random.py
"""
from mapf_demo import run_demo

if __name__ == "__main__":
    run_demo("lacam_random", [(0.0, 0.0)], spaces=(True,), inflate=lambda radius: 1, max_iterations=None)
