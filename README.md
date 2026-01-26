# Path Planning Repo

## TLDR
Fork from the AI winter python-motion-planning v2.0 repo to create this one [https://github.com/ai-winter/python_motion_planning]. 

Create a custom PRM and graphsampler functions.

Graph sampler
- Update the collision checking with a digital differential analyzer (DDA) to check if the line of sight or edge between two points overlays with any obstacle tiles
- Optimize generateRandomNodes functions with numpy and add an option to generate grid nodes as well
- Create a generate_roadmap (based on PRM) and generate_planar_map functions (no overlapping edges)

Create custom visualization for 2D & 3D to show the roadmaps from the algorithms or graph sampler


## Installation
To recreate the conda environment [linux]
```
conda env create -f environment.yml
conda activate path_planning
pip install poetry
python -m pip install .
```

To recreate the conda environment [window/mac]
```
conda create -n "path_planning" python=3.12 ipython
conda activate "path_planning"
conda install poetry
pip install poetry
python -m pip install .
```

## Script
Generating sample map in 2D and 3D grid space
```
python -m scripts.run_make_environments
```

Running our custom Probabilitis Road Map (PRM) algorithm on the sample maps
```
python -m scripts.run_prm
```

Running our custom Rapidly Exploring Random Graphs (RRG) algorithm on the sample maps
```
python -m scripts.run_rrg
```

Running our custom graph sampler to generate road map and planar map on the sample maps
```
python scripts/run_graph_sampler.py
```