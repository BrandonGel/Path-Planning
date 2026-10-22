from python_motion_planning.common import Node as base_node
import numpy as np


class Node(base_node):
    """Roadmap node: upstream ``Node`` plus a ``neighbors`` list.
    """

    def __init__(self, current: tuple, parent: tuple = None, g: float = 0, h: float = 0) -> None:
        self._current = current
        self.parent = parent
        self._g = g
        self._h = h
        self.neighbors = []
        if parent is not None and len(current) != len(parent):
            raise ValueError(
                "The dimension of current " + str(current) + " and parent " + str(parent) + " must be the same."
            )

    def link(self, node):
        return Node(self._current, node._current, node._g, self._h)
