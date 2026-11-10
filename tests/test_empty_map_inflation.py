"""An obstacle-free map must not grow phantom inflation cells.

scipy's ``distance_transform_edt`` is undefined when the free-space mask has
no background pixel; it returns distances to a phantom point at index (-1, 0),
which used to turn the cells next to the origin into INFLATION cells.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from python_motion_planning.common import TYPES  # noqa: E402

from path_planning.common.environment.map.graph_sampler import GraphSampler  # noqa: E402


def _empty_map(size=16.0):
    return GraphSampler(bounds=[[0.0, size], [0.0, size]], resolution=1.0, start=[], goal=[],
                        use_discrete_space=True)


def test_empty_map_esdf_is_uniform_and_far():
    m = _empty_map()
    e = m.esdf
    assert np.isfinite(e).all()
    assert np.allclose(e, e.flat[0])
    assert e.min() >= 16.0 * np.sqrt(2) - 1e-6


@pytest.mark.parametrize("radius", [0.5, 1.0, 0.3 + np.sqrt(2) / 2, 3.0])
def test_empty_map_has_no_inflation_cells(radius):
    m = _empty_map()
    m.inflate_obstacles(radius=radius)
    assert not np.any(m.type_map.data == TYPES.INFLATION)
    assert m.min_wall_distance((1.5, 1.5)) > 10.0


def test_map_with_obstacle_keeps_real_esdf():
    m = _empty_map()
    m.set_obstacles(np.array([[5, 5]], dtype=np.int64))
    e = m.esdf
    assert e[5, 5] <= 0.0
    assert np.isclose(e[5, 6], 1.0) and np.isclose(e[6, 6], np.sqrt(2))
    m.inflate_obstacles(radius=1.0)
    inflated = np.argwhere(m.type_map.data == TYPES.INFLATION)
    assert len(inflated) > 0
    assert np.all(np.abs(inflated - [5, 5]).max(axis=1) <= 1)
