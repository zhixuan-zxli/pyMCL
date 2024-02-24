from typing import List
import meshio
import numpy as np
from scipy.sparse import csr_matrix
from matplotlib.pyplot import triplot, axis

class Mesh:
    def __init__(self) -> None:
        self.point = np.zeros((0, 4), dtype=np.float64)
        self.edge = np.zeros((0, 3), dtype=np.uint32)
        self.tri = np.zeros((0, 4), dtype=np.uint32)
        self.tet = np.zeros((0, 5), dtype=np.uint32)

    def load(self, mesh_name: str) -> None:
        """
        Load a GMSH mesh. 
        """
        msh = meshio.read(mesh_name)
        # 1. read the points
        self.point = msh.points
        if np.all(self.point[:,-1] == 0.): 
            self.point = self.point[:,:-1] # Remove the z coordinates if this is a planar mesh
        self.point = np.hstack((self.point, np.zeros((self.point.shape[0], 1)))) # append the tags

        # 2. read the higher-dimensional entities
        num_cells = len(msh.cells)
        has_physical = "gmsh:physical" in msh.cell_data

        for cid in range(num_cells):
            cell = msh.cells[cid]
            if has_physical:
                data = msh.cell_data["gmsh:physical"][cid].astype(np.uint32).reshape(-1, 1) # reshape to a column vector
            else:
                data = np.zeros((cell.data.shape[0], 1), dtype=np.uint32)
            if cell.type == "vertex" and data != None:
                self.point[cell.data, -1] = data
            elif cell.type == "line":
                self.edge = np.vstack((self.edge, np.hstack((cell.data.astype(np.uint32), data))))
            elif cell.type == "triangle":
                self.tri = np.vstack((self.tri, np.hstack((cell.data.astype(np.uint32), data))))
            else:
                raise RuntimeError("Unrecognized cell type. ")

    def view(self, sub_dim: int, tags: List[int]) -> "Mesh":
        Np = self.point.shape[0]
        submesh = Mesh()
        if sub_dim == 0:
            # extract the points
            point_flag = np.zeros((Np,), dtype=np.bool8)
            for t in tags:
                point_flag[self.point[:,-1] == t] = True
            submesh.point_map = np.arange(0, Np)[point_flag]
            submesh.point = self.point[point_flag, :]
        elif sub_dim == 1:
            # extract the edges
            edge_flag = np.zeros((self.edge.shape[0],), dtype=np.bool8)
            for t in tags:
                edge_flag[self.edge[:,-1] == t] = True
            submesh.edge_map = np.arange(0, self.edge.shape[0], dtype=np.uint32)[edge_flag]
            submesh.edge = self.edge[edge_flag, :]
            # extract the points
            point_flag = np.zeros((Np,), dtype=np.bool8)
            point_flag[submesh.edge[:,0]] = True
            point_flag[submesh.edge[:,1]] = True
            submesh.point_map = np.arange(0, Np, dtype=np.uint32)[point_flag]
            submesh.point = self.point[point_flag]
            # finally fix the edge indices
            inv_point_map = np.cumsum(point_flag) - 1
            submesh.edge[:,0] = inv_point_map[submesh.edge[:,0]]
            submesh.edge[:,1] = inv_point_map[submesh.edge[:,1]]
        else:
            raise RuntimeError("Submesh of dimension {} not implemented".format(sub_dim))
        return submesh
    
    def refine(self) -> "Mesh":
        fine_mesh = Mesh()
        if self.tri.shape[0] > 0:
            # identify the new nodes
            Np = self.point.shape[0]
            Nt = self.tri.shape[0]
            tri_edges = self.tri[:, [0,1,1,2,2,0]].reshape(-1, 3, 2)
            S = csr_matrix((np.ones((3*Nt,), dtype=np.uint32), 
                        (np.min(tri_edges, axis=2).flatten(), np.max(tri_edges, axis=2).flatten())), 
                        shape=(Np, Np))
            assert(S.nnz == Np + Nt - 1) # check using Euler's formula
            S.data = np.arange(S.nnz, dtype=S.dtype) + Np
            row, col = S.nonzero()
            # set up the new nodes
            new_point = 0.5 * (self.point[row,:] + self.point[col,:])
            new_point[:,-1] = 0.
            fine_mesh.point = np.vstack((self.point, new_point))
            # set up the new edges
            mid_edge = S[np.min(self.edge[:,0:2], axis=1), np.max(self.edge[:,0:2], axis=1)].reshape(-1, 1)
            fine_mesh.edge = np.block([[self.edge[:,[0]], mid_edge, self.edge[:,[2]]], [mid_edge, self.edge[:,[1]], self.edge[:,[2]]]])
            # set up the new triangles
            mid_tri = np.zeros((Nt, 3), dtype=np.uint32)
            mid_tri[:,0] = S[np.min(self.tri[:,0:2], axis=1), np.max(self.tri[:,0:2], axis=1)]
            mid_tri[:,1] = S[np.min(self.tri[:,1:3], axis=1), np.max(self.tri[:,1:3], axis=1)]
            mid_tri[:,2] = S[np.min(self.tri[:,[0,2]], axis=1), np.max(self.tri[:,[0,2]], axis=1)]
            fine_mesh.tri = np.block([[self.tri[:,[0]], mid_tri[:,[0]], mid_tri[:,[2]], self.tri[:,[3]]], 
                                      [mid_tri[:,[0]], self.tri[:,[1]], mid_tri[:,[1]], self.tri[:,[3]]], 
                                      [mid_tri[:,[1]], self.tri[:,[2]], mid_tri[:,[2]], self.tri[:,[3]]], 
                                      [mid_tri[:,[0]], mid_tri[:,[1]], mid_tri[:,[2]], self.tri[:,[3]]]])
        elif self.edge.shape[0] > 0:
            raise RuntimeError("Refine 1D mesh to be implemented. ")
        else:
            raise RuntimeError("There is nothing to refine. ")
        return fine_mesh
    
    def draw(self) -> None:
        if self.tet.shape[0] > 0:
            pass
        elif self.tri.shape[0] > 0:
            triplot(self.point[:,0], self.point[:,1], self.tri[:,:-1])
        elif self.edge.shape[0] > 0:
            pass
        axis("equal")

