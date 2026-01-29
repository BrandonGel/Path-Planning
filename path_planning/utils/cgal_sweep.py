from CGAL.CGAL_Kernel import Point_3, Segment_3, squared_distance, Point_2, Segment_2
import numpy as np

class CGAL_Sweep:
    def __init__(self,record_sweep: bool = True,use_exact_collision_check: bool = True):
        self.reset()
        self.record_sweep = record_sweep
        self.use_exact_collision_check = use_exact_collision_check

    def reset(self):
        self.Point_type = None
        self.Segment_type = None
        self.vertices = []
        self.edges = []
        self.edge_indices = {}
        self.overlapping_sweep = {}

    def set_graph(self,vertices: list[tuple[float,float]], edges: list[tuple[int,int]]):
        self.reset()
        dim = vertices[0].__len__()
        if dim == 2:
            self.Point_type = Point_2
            self.Segment_type = Segment_2
        elif dim == 3:
            self.Point_type = Point_3
            self.Segment_type = Segment_3

        for vertex in vertices:
            p_pt = self.Point_type(*vertex)
            self.vertices.append(p_pt)

        for edge in edges:
            src,tgt = edge
            a_pt = self.Point_type(*vertices[src])
            b_pt = self.Point_type(*vertices[tgt])
            self.edges.append(self.Segment_type(a_pt, b_pt))
            self.edge_indices[len(self.edges)-1] = (src,tgt)

    def overlapping_graph_elements_cgal(self, u: tuple[float,float], v: tuple[float,float], r: float):
        if self.record_sweep and (u,v,r) in self.overlapping_sweep:
            overlapping_vertices, overlapping_edges, overlapping_start_vertices = self.overlapping_sweep[u,v,r]
            return overlapping_vertices, overlapping_edges, overlapping_start_vertices
            
        u_pt = self.Point_type(*u)
        v_pt = self.Point_type(*v)
        traversal_seg = self.Segment_type(u_pt, v_pt)

        # --- Vertex overlap ---
        overlapping_vertices = set()
        overlapping_start_vertices = set()
        for ii, p_pt in enumerate(self.vertices):
            if squared_distance(p_pt, traversal_seg)**0.5 < r:
                overlapping_vertices.add(ii)
                if 0.0 <= squared_distance(p_pt, u_pt)**0.5 < r:
                    overlapping_start_vertices.add(ii)

        # --- Edge overlap ---
        overlapping_edges = set(self.edge_indices[ii]  for ii, edge in enumerate(self.edges) if squared_distance(traversal_seg, edge)**0.5 < r )

        
        if self.use_exact_collision_check:
            crossing_edges = set()
            for edge in overlapping_edges:
                src,tgt = edge
                if src not in overlapping_vertices:
                    crossing_edges.add(edge)
                if tgt not in overlapping_vertices:
                    crossing_edges.add(edge)

            remove_edges = set()
            remove_vertices = set()
            for edge in crossing_edges:
                src,tgt = edge
                a_pt = self.vertices[src]
                b_pt = self.vertices[tgt]

                u_to_v = v_pt-u_pt
                a_to_b = b_pt-a_pt

                # --- Edge 1 ---
                # a -> b and u -> v 
                ro1 = a_pt-u_pt
                vel1 = a_to_b-u_to_v
                tmin1 = np.clip(-np.dot(ro1,vel1)/(np.dot(vel1,vel1)+1e-10),0.0,1.0)
                vec1 = ro1 + vel1*tmin1
                if vec1.squared_length()**0.5 > r:
                    remove_edges.add(edge)
                    if tgt in overlapping_vertices:
                        remove_vertices.add(tgt)

                # --- Edge 2 ---
                # b -> a and u -> v 
                ro2 = b_pt-u_pt
                vel2 = -a_to_b-u_to_v
                tmin2 = np.clip(-np.dot(ro2,vel2)/(np.dot(vel2,vel2)+1e-10),0.0,1.0)
                vec2 = ro2 + vel2*tmin2
                if vec2.squared_length()**0.5 > r:
                    remove_edges.add(edge)
                    if src in overlapping_vertices:
                        remove_vertices.add(src)

            overlapping_vertices = overlapping_vertices - remove_vertices
            overlapping_edges = overlapping_edges - remove_edges

        if self.record_sweep:
            self.overlapping_sweep[u,v,r] = (overlapping_vertices, overlapping_edges, overlapping_start_vertices)
        return overlapping_vertices, overlapping_edges, overlapping_start_vertices
