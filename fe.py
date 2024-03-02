from typing import List, Optional
import numpy as np
from mesh import *

class FiniteElement:

    tdim: int
    rdim: int
    num_dof_per_elem: int
    num_dof_per_dim: np.ndarray
    num_dof: int
    mesh: Mesh

class TriP2(FiniteElement):

    tdim: int = 2
    rdim: int = 1
    num_dof_per_elem: int = 6
    name: str = "Triangular P2"

    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh
        assert(mesh.tri.shape[0] > 0)
        edge_table = mesh.get_edge_table()
        self.num_dof_per_dim = np.array([mesh.point.shape[0], edge_table.shape[0]], dtype=np.uint32)
        self.num_dof = np.sum(self.num_dof_per_dim)
        # build cell dofs
        self.cell_dof = np.zeros((mesh.tri.shape[0], self.num_dof_per_elem), dtype=np.uint32)
        self.cell_dof[:, :3] = mesh.tri
        temp = mesh.tri[:, [0,1,1,2,2,0]].reshape(-1, 3, 2)
        temp = np.stack((np.min(temp, axis=2), np.max(temp, axis=2)), axis=2).reshape(-1, 2)
        temp = np.array([
            edge_table[r.tobytes()] for r in temp
        ], dtype=np.uint32)
        self.cell_dof[:, 3:] = temp.reshape(-1, 3)
        # build the facet dofs
        self.facet_dof = np.zeros((mesh.edge.shape[0], 3), dtype=np.uint32)
        self.facet_dof[:, :2] = mesh.edge
        temp = np.vstack((np.min(mesh.edge, axis=1), np.max(mesh.edge, axis=1))).T
        self.facet_dof[:, 2] = np.array([
            edge_table[r.tobytes()] for r in temp
        ], dtype=np.uint32)

    # for imposing Dirichlet condition
    def getFacetDof(self, sub_ids: Optional[List[int]] = None) -> np.ndarray:
        if sub_ids == None:
            dofs = self.facet_dof
        else:
            flag = np.zeros((self.facet_dof.shape[0], ), np.bool8)
            for t in sub_ids:
                flag[self.mesh.edge_tag == t] = True
            dofs = self.facet_dof[flag]
        return dofs.unique()

    @classmethod
    def _eval_basis_2(basis_id:int, qpts: np.ndarray) -> np.ndarray:
        assert(basis_id < TriP2.num_dof_per_elem)
        x = qpts[0, :]
        y = qpts[1, :]
        if basis_id == 0:
            return 2.0*x**2 - 3.0*x + 1.0 + 2.0*y**2 - 3.0*y + 4.0*x*y
        if basis_id == 1:
            return 2.0*x*(x-1.0/2)
        if basis_id == 2:
            return 2.0*y*(y-1.0/2)
        if basis_id == 3:
            return -4.0*x*(x+y-1)
        if basis_id == 4:
            return 4.0*x*y
        if basis_id == 5:
            return -4.0*y*(x+y-1)
    
    @classmethod
    def _eval_grad_2(basis_id:int, qpts: np.ndarray) -> np.ndarray:
        assert(basis_id < TriP2.num_dof_per_elem)
        x = qpts[0, :]
        y = qpts[1, :]
        if basis_id == 0:
            return np.vstack((4.0*x+4.0*y-3.0, 4.0*x+4.0*y-3.0))
        if basis_id == 1:
            return np.vstack((4.0*x-1.0, 0.0*y))
        if basis_id == 2:
            return np.vstack((0.0*x, 4.0*y-1.0))
        if basis_id == 3:
            return np.vstack((-8.0*x-4.0*y+4.0, -4.0*x))
        if basis_id == 4:
            return np.vstack((4.0*y, 4.0*x))
        if basis_id == 5:
            return np.vstack((-4.0*y, -4.0*x-8.0*y+4.0))
        
