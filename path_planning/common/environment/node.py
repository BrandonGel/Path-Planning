from python_motion_planning.common import  Node as base_node
import numpy as np

class Node(base_node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = None
        self.neighbors = []
        # if self.current is not None:
        #     self.current_round =  tuple(round(_,4) for _ in self.current)
        # else:
        #     self.current_round = None
        # if self.parent is not None:
        #     self.parent_round = tuple(round(_,4) for _ in self.parent)
        # else:
        #     self.parent_round = None


    def link(self, node):
        new_node = Node(node.current,self.current,node.g+self.g,node.h)
        new_node.neighbors = node.neighbors
        return new_node

    # def __str__(self) -> str:
    #     return "Node({}, {}, {}, {})".format(self.current_round, self.parent_round, self._g, self._h)