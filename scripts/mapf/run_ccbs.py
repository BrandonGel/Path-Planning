"""
Run CCBS on the 2d map in discrete and continuous space. CCBS requires radius > 0; velocity 0 means every
move takes 1 time unit. Cases are (velocity, radius).
python scripts/mapf/run_ccbs.py
"""
from mapf_demo import run_demo

if __name__ == "__main__":
    run_demo("ccbs", [(0.0, 1.0), (1.0, 1.0), (0.0, 2.0)])
