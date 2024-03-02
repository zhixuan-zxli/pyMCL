# from typing import Optional
import numpy as np
from mesh import *

class FiniteElement:

    tdim: int
    rdim: int
    num_dof_per_elem: int
    num_dof_per_dim: np.ndarray
    num_dof: int
    mesh: Mesh
    name: str
    # cell_dof

    def getDof(self, dim: int, sub_ids = None) -> np.ndarray:
        if dim == 0:
            if sub_ids == None:
                return np.arange(self.num_dof_per_dim[0], dtype=np.uint32)
            else:
                flag = np.zeros((self.mesh.point.shape[0], ), np.bool8)
                for i in sub_ids:
                    flag[self.mesh.point_tag == i] = True
                return np.nonzero(flag)[0].astype(np.uint32)
        if sub_ids == None:
            return self.cell_dof[dim][:, :-1].unique()
        flag = np.zeros((self.cell_dof[dim].shape[0], ), np.bool8)
        for t in sub_ids:
            flag[self.cell_dof[dim][:, -1] == t] = True
        return self.cell_dof[dim][flag, :-1].unique()

class EdgeP2(FiniteElement):
    
    tdim: int = 1
    rdim: int = 1
    num_dof_per_elem: int = 3
    name: str = "Edge P2"
    trace_type = None

class TriP2(FiniteElement):

    tdim: int = 2
    rdim: int = 1
    num_dof_per_elem: int = 6
    name: str = "Triangular P2"
    trace_type = EdgeP2

    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh
        assert(mesh.cell[2].shape[0] > 0)
        edge_table = mesh.get_entities(1)
        self.num_dof_per_dim = np.array([mesh.point.shape[0], len(edge_table)], dtype=np.uint32)
        self.num_dof = np.sum(self.num_dof_per_dim)
        self.cell_dof = [None] * 3
        # build cell dofs
        self.cell_dof[2] = np.zeros((mesh.cell[2].shape[0], self.num_dof_per_elem+1), dtype=np.uint32)
        self.cell_dof[2][:, :3] = mesh.cell[2][:, :3]
        temp = mesh.cell[2][:, [0,1,1,2,2,0]].reshape(-1, 3, 2)
        temp = np.stack((np.min(temp, axis=2), np.max(temp, axis=2)), axis=2).reshape(-1, 2)
        temp = np.array([
            edge_table[r.tobytes()] for r in temp
        ], dtype=np.uint32)
        self.cell_dof[2][:, 3:-1] = temp.reshape(-1, 3)
        self.cell_dof[2][:, -1] = mesh.cell[2][:, -1]
        # build the facet dofs
        self.cell_dof[1] = np.zeros((mesh.cell[1].shape[0], self.trace_type.num_dof_per_elem + 1), dtype=np.uint32)
        self.cell_dof[1][:, :2] = mesh.cell[1][:, :2]
        temp = np.vstack((np.min(mesh.cell[1][:, :-1], axis=1), np.max(mesh.cell[1][:, :-1], axis=1))).T
        self.cell_dof[1][:, 2] = np.array([
            edge_table[r.tobytes()] for r in temp
        ], dtype=np.uint32)
        self.cell_dof[1][:, -1] = mesh.cell[1][:, -1]

    @classmethod
    def _eval_basis(basis_id:int, qpts: np.ndarray) -> np.ndarray:
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
    def _eval_grad(basis_id:int, qpts: np.ndarray) -> np.ndarray:
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
        
