from python_motion_planning.common.utils.geometry import Geometry
from typing import List, Tuple
import numpy as np
import math


class GeometryNumpy(Geometry):
    """
    Geometry tools class
    """

    @staticmethod
    def dist(p1: tuple | np.ndarray, p2: tuple | np.ndarray, type: str = 'Euclidean') -> float:
        """
        Calculate the distance between two points

        Args:
            p1: First point(s)
            p2: Second point(s)
            type: Type of distance calculation, either 'Euclidean' or 'Manhattan'

        Returns:
            dist: Distance between the two points
        """
        if isinstance(p1, tuple) and isinstance(p2, tuple):
            return Geometry.dist(p1, p2, type)
        elif isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
            if len(p1) != len(p2):
                raise ValueError("Points must have the same number of dimensions")
            if type == 'Euclidean':
                return np.linalg.norm(p1 - p2, axis=-1, ord=2)
            elif type == 'Manhattan':
                return np.linalg.norm(p1 - p2, axis=-1, ord=1)
            else:
                raise ValueError("Invalid distance type")
        else:
            raise ValueError("Invalid point type")

        
    @staticmethod
    def add_orient_to_2d_path(path: List[Tuple[float, float]] | np.ndarray) -> List[Tuple[float, float, float] | np.ndarray]:
        """
        Add orientation information to a 2D point path. Each point in the path has a third value representing the angle (in radians) between the current point and the next point.

        Args:
            path: a list of 2D points
            
        Returns:
            new_path: a list of 2D poses
        """
        if isinstance(path, List[Tuple[float, float]]):
            return Geometry.add_orient_to_2d_path(path)
        elif isinstance(path, np.ndarray):
            if len(path) < 2:
                return np.hstack((path, np.zeros((len(path), 1))))
            
            path_with_orient = np.zeros((len(path), 3))
            path_with_orient[:, :2] = path

            # Get the orientation of the path
            diff = path[1:] - path[:-1]
            orient = np.arctan2(diff[:, 1], diff[:, 0])

            # Add the orientation to the path
            path_with_orient[:len(orient), 2] = orient

            # Add the last orientation to the path
            path_with_orient[-1,-1] = orient[-1]

            return path_with_orient
        else:
            raise ValueError("Invalid path type")
