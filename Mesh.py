from typing import List
import meshio
import numpy as np
from scipy.sparse import csr_matrix

class Mesh:
    def load(self, mesh_name: str) -> None:
        """
        Load a GMSH mesh. 
        """
        msh = meshio.read(mesh_name)
        self.Np = msh.points.shape[0]
        # 1. read the points
        self.point = msh.points
        if np.all(self.point[:,-1] == 0.): 
            self.point = self.point[:,:-1] # Remove the z coordinates if this is a planar mesh
        self.point = np.hstack((self.point, np.zeros((self.Np, 1)))) # append the tags

        # 2. read the higher-dimensional entities
        num_cells = len(msh.cells)
        has_physical = "gmsh:physical" in msh.cell_data
        self.edge = np.zeros((0, 3), dtype=np.uint32)
        self.tri = np.zeros((0, 4), dtype=np.uint32)

        for cid in range(num_cells):
            cell = msh.cells[cid]
            if has_physical:
                data = msh.cell_data["gmsh:physical"][cid].reshape(-1, 1) # reshape to a column vector
            else:
                data = np.zeros((cell.data.shape[0], 1), dtype=np.uint32)
            if cell.type == "vertex" and data != None:
                self.point[cell.data, -1] = data
            elif cell.type == "line":
                self.edge = np.vstack((self.edge, np.hstack((cell.data, data))))
            elif cell.type == "triangle":
                self.tri = np.vstack((self.tri, np.hstack((cell.data, data))))
            else:
                raise RuntimeError("Unrecognized cell type. ")
            
    # def isP2Mesh(self) -> bool:
    #     return len(self.Np) == 2

    # def buildP2Mesh(self) -> None:
    #     """
    #     Build the P2 mesh. 
    #     """
    #     assert(not self.isP2Mesh())
    #     Np = self.Np[0]
    #     T = self.tri
    #     Nt = T.shape[0]
    #     # figure out the new dofs
    #     tri_edges = np.block([ [T[:,[0]], T[:,[1]]], [T[:,[2]], T[:,[3]]], [T[:,[2]], T[:,[0]]] ])
    #     S = csr_matrix((np.ones((3*Nt,)), (np.min(tri_edges, axis=1), np.max(tri_edges, axis=1))), shape=(Np, Np))
    #     S.data = np.arange(S.nnz, dtype=S.dtype) + (Np+1)
    #     row, col = S.nonzero()
    #     # build new nodes
    #     new_points = 0.5 * (self.point[row, :] + self.point[col, :])
    #     new_points[:, -1] = 0. # reset the tags
    #     self.point = np.vstack((self.point, new_points))
    #     self.Np += [self.point.shape[0]]
    #     # update edges
    #     self.edge = np.hstack((self.edge, self.edge[:, [-1]]))
    #     self.edge[:,-2] = S[np.min(self.edge[:,0:2], axis=1), np.max(self.edge[:,0:2], axis=1)]
    #     # update the triangles
    #     self.tri = np.hstack((self.tri[:,0:3], np.zeros((Nt, 3), dtype=self.tri.dtype), self.tri[:,[-1]]))
    #     self.tri[:,3] = S[np.min(self.tri[:,0:2], axis=1), np.max(self.tri[:,0:2], axis=1)]
    #     self.tri[:,4] = S[np.min(self.tri[:,1:3], axis=1), np.max(self.tri[:,1:3], axis=1)]
    #     self.tri[:,5] = S[np.min(self.tri[:,[0,2]], axis=1), np.max(self.tri[:,[0,2]], axis=1)]
    #     # todo: sort nodes?

    def view(self, sub_dim: int, tags: List[int]) -> "Mesh":
        submesh = Mesh()
        if sub_dim == 0:
            # extract the points
            point_flag = np.zeros((self.Np,), dtype=np.bool8)
            for t in tags:
                point_flag[self.point[:,-1] == t] = True
            submesh.point_map = np.arange(0, self.Np)[point_flag]
            submesh.point = self.point[point_flag, :]
        elif sub_dim == 1:
            # extract the edges
            edge_flag = np.zeros((self.edge.shape[0],), dtype=np.bool8)
            for t in tags:
                edge_flag[self.edge[:,-1] == t] = True
            submesh.edge_map = np.arange(0, self.edge.shape[0])[edge_flag]
            submesh.edge = self.edge[edge_flag, :]
            # extract the points
            point_flag = np.zeros((self.point.shape[0],), dtype=np.bool8)
            point_flag[submesh.edge[:,0]] = True
            point_flag[submesh.edge[:,1]] = True
            submesh.point_map = np.arange(0, self.Np)[point_flag]
            submesh.point = self.point[point_flag]
            # finally fix the edge indices
            inv_point_map = np.cumsum(point_flag) - 1
            submesh.edge[:,0] = inv_point_map[submesh.edge[:,0]]
            submesh.edge[:,1] = inv_point_map[submesh.edge[:,1]]
        else:
            raise RuntimeError("Submesh of dimension {} not implemented".format(sub_dim))
        return submesh


