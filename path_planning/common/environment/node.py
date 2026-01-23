from python_motion_planning.common import  Node as base_node
import numpy as np

class Node(base_node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neighbors = []

    def link(self, node):
        new_node = Node(node.current,self.current,node.g+self.g,node.h)
        new_node.neighbors = node.neighbors
        return new_node

# from __future__ import annotations
# import numpy as np
# class Node(object):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.neighbors = []

#     def link(self, node):
#         new_node = Node(node.current,self.current,node.g+self.g,node.h)
#         new_node.neighbors = node.neighbors
#         return new_node

#     # g: cost to reach the node
#     def __init__(self, current: tuple, parent: tuple = None, g: float = 0, h: float = 0) -> None:
#         self._current = current
#         self.parent = parent

#         if self.parent is not None and len(self.current) != len(self.parent):
#             raise ValueError("The dimension of current " + str(self.current) + " and parent " + str(self.parent) + " must be the same.")
    
#     def __eq__(self, node: Node) -> bool:
#         if not isinstance(node, Node):
#             return False
#         return self._current == node._current
    
#     def __ne__(self, node: Node) -> bool:
#         return not self.__eq__(node)


#     def __hash__(self) -> int:
#         return hash(self._current)

#     def __str__(self) -> str:
#         return "Node({}, {})".format(self._current, self.parent)

#     def __repr__(self) -> str:
#         return self.__str__()

#     def __len__(self) -> int:
#         return len(self.current)

#     @property
#     def current(self) -> tuple:
#         return self._current


#     @property
#     def dim(self) -> int:
#         return len(self.current)