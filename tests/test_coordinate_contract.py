"""Coordinate contract: YAML obstacles = world centers; agent start/goal = world coords."""

import unittest
import numpy as np

try:
    from path_planning.common.environment.map.graph_sampler import GraphSampler
    from path_planning.utils.util import (
        agents_yaml_to_roadmap_frame,
        obstacles_world_to_grid,
        validate_obstacle_map_config,
    )

    _HAS_SAMPLER = True
except ModuleNotFoundError:
    _HAS_SAMPLER = False


def _discrete_cell_from_world_list(m, world_list):
    """Match ``agents_yaml_to_roadmap_frame`` / ``world_to_map(..., discrete=True)`` output."""
    t = m.world_to_map(tuple(float(x) for x in world_list), discrete=True)
    return tuple(int(x) for x in np.asarray(t).reshape(-1))


@unittest.skipUnless(_HAS_SAMPLER, "GraphSampler/CGAL not available in this environment")
class TestCoordinateContract(unittest.TestCase):
    def test_agents_yaml_to_grid_roundtrip_2d(self):
        bounds = [[0.0, 8.0], [0.0, 8.0]]
        resolution = 2.0
        m = GraphSampler(
            bounds=bounds,
            resolution=resolution,
            start=[],
            goal=[],
            use_discrete_space=True,
        )
        cell_s = (1, 2)
        cell_g = (3, 4)
        w_s = m.map_to_world(cell_s)
        w_g = m.map_to_world(cell_g)
        start_list = [float(x) for x in np.asarray(w_s).reshape(-1)]
        goal_list = [float(x) for x in np.asarray(w_g).reshape(-1)]
        agents = [
            {
                "name": "a0",
                "start": start_list,
                "goal": goal_list,
            }
        ]
        rt = agents_yaml_to_roadmap_frame(m, agents)
        # map_to_world(cell) and world_to_map are not always exact inverses (e.g. cell
        # centers); assert the helper matches the grid’s own discrete projection.
        self.assertEqual(rt[0]["start"], _discrete_cell_from_world_list(m, start_list))
        self.assertEqual(rt[0]["goal"], _discrete_cell_from_world_list(m, goal_list))

    def test_resolution_scaling_dimensions(self):
        bounds = [[0.0, 32.0], [0.0, 32.0]]
        for res in (1.0, 0.5, 2.0):
            m = GraphSampler(
                bounds=bounds,
                resolution=res,
                start=[],
                goal=[],
                use_discrete_space=True,
            )
            self.assertEqual(m.shape[0], int((bounds[0][1] - bounds[0][0]) / res))
            self.assertEqual(m.shape[1], int((bounds[1][1] - bounds[1][0]) / res))

    def test_obstacle_world_to_grid_contains_center_cell(self):
        bounds = [[0.0, 8.0], [0.0, 8.0]]
        for res in (1.0, 0.5, 2.0):
            m = GraphSampler(
                bounds=bounds,
                resolution=res,
                start=[],
                goal=[],
                use_discrete_space=True,
            )
            world_center = [3.0, 3.0]
            center_cell = tuple(m.world_to_map(tuple(world_center), discrete=True))
            obs = obstacles_world_to_grid(m, [world_center], obs_size=2.0)
            obs_set = {tuple(int(v) for v in row) for row in obs.tolist()}
            self.assertIn(tuple(int(v) for v in center_cell), obs_set)

    def test_obstacle_zero_size_maps_single_cell(self):
        m = GraphSampler(
            bounds=[[0.0, 8.0], [0.0, 8.0]],
            resolution=1.0,
            start=[],
            goal=[],
            use_discrete_space=True,
        )
        world_center = [4.0, 5.0]
        center_cell = tuple(int(v) for v in m.world_to_map(tuple(world_center), discrete=True))
        obs = obstacles_world_to_grid(m, [world_center], obs_size=0.0)
        self.assertEqual(obs.shape[0], 1)
        self.assertEqual(tuple(int(v) for v in obs[0].tolist()), center_cell)


    def test_validate_rejects_unaligned_obstacle(self):
        m = GraphSampler(
            bounds=[[0.0, 8.0], [0.0, 8.0]],
            resolution=1.0,
            start=[],
            goal=[],
            use_discrete_space=True,
        )
        # 0.3 does not snap to a grid cell with resolution=1 (frac=0.3)
        with self.assertRaises(ValueError):
            validate_obstacle_map_config(m, [[0.3, 1.0]], obs_size=1.0)
        # Integer world coords always snap cleanly
        validate_obstacle_map_config(m, [[1.0, 2.0]], obs_size=1.0)

    def test_validate_accepts_aligned_obstacles_nonunit_resolution(self):
        m = GraphSampler(
            bounds=[[0.0, 8.0], [0.0, 8.0]],
            resolution=0.5,
            start=[],
            goal=[],
            use_discrete_space=True,
        )
        # Multiples of 0.5 align; 0.3 does not
        validate_obstacle_map_config(m, [[0.0, 0.5], [1.0, 1.5]], obs_size=0.5)
        with self.assertRaises(ValueError):
            validate_obstacle_map_config(m, [[0.3, 0.0]], obs_size=0.5)


if __name__ == "__main__":
    unittest.main()
