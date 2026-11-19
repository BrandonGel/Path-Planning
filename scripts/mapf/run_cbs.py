"""
Run CBS on the 2d map with agent radii 0.0, 1.0 and 2.0 (unit time steps), in discrete and continuous space.
python scripts/mapf/run_cbs.py
"""
from mapf_demo import run_demo

if __name__ == "__main__":
    run_demo("cbs", [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)])
