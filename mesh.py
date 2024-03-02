from typing import List
import meshio
import numpy as np
from scipy.sparse import csr_matrix
from matplotlib import pyplot

class Mesh:
    def __init__(self) -> None:
        self.point = None
        self.point_tag = None
        self.edge = np.zeros((0, 2), dtype=np.uint32)
        self.tri = np.zeros((0, 3), dtype=np.uint32)
        # self.quad = np.zeros((0, 4), dtype=np.uint32)
        # self.tet = np.zeros((0, 4), dtype=np.uint32)
        self.edge_tag = np.zeros((0, ), dtype=np.uint32)
        self.tri_tag = np.zeros((0, ), dtype=np.uint32)

    def load(self, mesh_name: str) -> None:
        """
        Load a GMSH mesh. 
        """
        msh = meshio.read(mesh_name)
        # 1. read the points
        self.point = msh.points
        if np.all(self.point[:,-1] == 0.): 
            self.point = self.point[:,:-1] # Remove the z coordinates if this is a planar mesh
        self.point_tag = np.zeros((self.point.shape[0],), dtype=np.uint32)

        # 2. read the higher-dimensional entities
        num_cells = len(msh.cells)
        has_physical = "gmsh:physical" in msh.cell_data

        for cid in range(num_cells):
            cell = msh.cells[cid]
            if has_physical:
                data = msh.cell_data["gmsh:physical"][cid]
            else:
                data = np.zeros((cell.data.shape[0],), dtype=np.uint32)
            if cell.type == "vertex" and data != None:
                self.point_tag[cell.data] = data
            elif cell.type == "line":
                self.edge = np.vstack((self.edge, cell.data))
                self.edge_tag = np.hstack((self.edge_tag, data))
            elif cell.type == "triangle":
                self.tri = np.vstack((self.tri, cell.data))
                self.tri_tag = np.hstack((self.tri_tag, data))
            else:
                raise RuntimeError("Unrecognized cell type. ")
            
    def get_edge_table(self) -> dict:
        if not hasattr(self, "edge_table"):
            temp = self.tri[:, [0,1,1,2,2,0]].reshape(-1, 3, 2)
            edge_table = np.stack((np.min(temp, axis=2), np.max(temp, axis=2)), axis=2).reshape(-1, 2)
            edge_table = np.unique(edge_table, axis=0)
            self.edge_table = {r.tobytes(): i for i, r in enumerate(edge_table)}
        return self.edge_table

    # def view(self, sub_dim: int, tags: List[int]) -> "Mesh":
    #     Np = self.point.shape[0]
    #     submesh = Mesh()
    #     if sub_dim == 0:
    #         # extract the points
    #         point_mask = np.zeros_like(self.point_tag, dtype=np.bool8)
    #         for t in tags:
    #             point_mask[self.point_tag == t] = True
    #         submesh.point_map = np.arange(0, Np)[point_mask]
    #         submesh.point = self.point[point_mask]
    #         submesh.point_tag = self.point_tag[point_mask]
    #     elif sub_dim == 1:
    #         # extract the edges
    #         edge_mask = np.zeros_like(self.edge_tag, dtype=np.bool8)
    #         for t in tags:
    #             edge_mask[self.edge_tag == t] = True
    #         submesh.edge_map = np.arange(0, self.edge.shape[0], dtype=np.uint32)[edge_mask]
    #         submesh.edge = self.edge[edge_mask, :]
    #         submesh.edge_tag = self.edge_tag[edge_mask, :]
    #         # extract the points
    #         point_mask = np.zeros((Np,), dtype=np.bool8)
    #         point_mask[submesh.edge.flatten()] = True
    #         submesh.point_map = np.arange(0, Np, dtype=np.uint32)[point_mask]
    #         submesh.point = self.point[point_mask]
    #         submesh.point_tag = self.point_tag[point_mask]
    #         # finally fix the edge indices
    #         inv_point_map = np.cumsum(point_mask) - 1
    #         submesh.edge = inv_point_map[submesh.edge.flatten()]
    #         submesh.edge = submesh.edge.reshape(-1, 2)
    #     else:
    #         raise NotImplementedError
    #     return submesh
    
    def draw(self) -> None:
        if self.tri.shape[0] > 0:
            pyplot.triplot(self.point[:,0], self.point[:,1], self.tri)
        elif self.edge.shape[0] > 0:
            pass
        pyplot.axis("equal")

