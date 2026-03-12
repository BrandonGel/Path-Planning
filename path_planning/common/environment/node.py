from python_motion_planning.common import  Node as base_node
import numpy as np

class Node(base_node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neighbors = []

    def link(self, node):
         return Node(self._current, node._current, node._g, self._h)