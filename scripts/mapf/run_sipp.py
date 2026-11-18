"""
Run SIPP on the 2d map in discrete and continuous space. Cases are (velocity, radius); radius 0: point
particles, velocity 0: every move takes 1 time unit.
python scripts/mapf/run_sipp.py
"""
from mapf_demo import run_demo

if __name__ == "__main__":
    run_demo("sipp", [(0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (1.0, 1.0)], to_roadmap_frame=True)
