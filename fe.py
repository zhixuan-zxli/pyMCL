from typing import Optional
import numpy as np
from mesh import Mesh

class Measure:
    def __init__(self, tdim: int, sub_id: Optional[int] = None) -> None:
        self.tdim = tdim
        self.sub_id = sub_id

class RefCell:
    tdim: int
    dx: float

class RefNode(RefCell):
    tdim: int = 0
    dx: float = 0.0

class RefLine(RefCell):
    tdim: int = 1
    dx: float = 1.0

class RefTri(RefCell):
    tdim: int = 2
    dx: float = 1.0/2

class FiniteElement:

    # class attributes
    ref_cell: RefCell
    tdim: int
    rdim: int
    degree: int
    num_dof_per_elem: int

    # finite element attributes
    num_copy: int
    num_dof_per_dim: np.ndarray
    num_dof: int
    mesh: Mesh
    # cell_dof

    def __init__(self, mesh: Mesh, num_copy: int = 1) -> None:
        self.mesh = mesh
        self.num_copy = num_copy

    def getCellDof(self, mea: Measure) -> np.ndarray:
        if mea.tdim == 0:
            if mea.sub_id == None:
                return np.arange(self.num_dof_per_dim[0], dtype=np.uint32)
            else:
                flag = np.zeros((self.mesh.point.shape[0], ), np.bool8)
                for i in mea.sub_id:
                    flag[self.mesh.point_tag == i] = True
                return np.nonzero(flag)[0].astype(np.uint32)
        if mea.sub_id == None:
            return self.cell_dof[mea.tdim][:, :-1] # remove the tag
        flag = np.zeros((self.cell_dof[mea.tdim].shape[0], ), np.bool8)
        for t in mea.sub_id:
            flag[self.cell_dof[mea.tdim][:, -1] == t] = True
        return self.cell_dof[mea.tdim][flag, :-1] # remove the tag
    
    # todo: collapse
    
class NodeElement(FiniteElement):

    ref_cell = RefNode
    tdim: int = 0
    rdim: int = 1
    degree: int = 0
    num_dof_per_elem: int = 1
    
    def __init__(self, mesh: Mesh, num_copy: int = 1) -> None:
        raise RuntimeError("Why are you initializing a node element?")

    @staticmethod
    def _eval_basis(basis_id: int, qpts: np.ndarray) -> np.ndarray:
        assert(basis_id == 0)
        return np.ones((1, qpts.shape[1]))
    
    @staticmethod
    def _eval_grad(basis_id: int, qpts: np.ndarray) -> np.ndarray:
        raise RuntimeError("Evalating gradient of a node element. ")


# =====================================================================
# Line elements    

class LineElement(FiniteElement):
    
    ref_cell = RefLine
    tdim: int = 1

class LineDG0(LineElement):

    rdim: int = 1
    degree: int = 0
    num_dof_per_elem: int = 1
    trace_type = [NodeElement]

    # todo <<<<

class LineP1(FiniteElement):

    rdim: int = 1
    degree: int = 1
    num_dof_per_elem: int = 2
    trace_type = [NodeElement]

    # todo <<<<<<<<

class LineP2(LineElement):
    
    rdim: int = 1
    degree: int = 2
    num_dof_per_elem: int = 3
    trace_type = [NodeElement]

    # todo <<<<<<<<

# =====================================================================
# Triangular elements

class TriElement(FiniteElement):
    
    ref_cell = RefTri
    tdim: int = 2

    def __init__(self, mesh: Mesh, num_copy: int = 1) -> None:
        super().__init__(mesh, num_copy)
        assert(mesh.cell[2].shape[0] > 0)
        self.cell_dof = [None] * 3

class TriDG0(TriElement):

    rdim: int = 1
    degree: int = 0
    num_dof_per_elem: int = 1
    trace_type = [NodeElement, LineDG0]

    def __init__(self, mesh: Mesh, num_copy: int = 1) -> None:
        Nt = mesh.cell[2].shape[0]
        self.num_dof_per_dim = np.array((0, 0, Nt), dtype=np.int64)
        self.num_dof = Nt
        # build cell dof
        self.cell_dof[2] = np.arange(Nt).reshape(-1, 1)
        # build facet dof
        # skip building facet dofs as it is not used

    @staticmethod
    def _eval_basis(basis_id: int, qpts: np.ndarray) -> np.ndarray:
        assert basis_id == 0
        return np.ones((1, qpts.shape[1]))
    
    @staticmethod
    def _eval_grad(basis_id: int, qpts: np.ndarray) -> np.ndarray: # rdim * tdim * num_quad
        return np.zeros((1, 2, qpts.shape[1]))

class TriP1(TriElement):

    rdim: int = 1
    degree: int = 1
    num_dof_per_elem: int = 3
    trace_type = [NodeElement, LineP1]

    def __init__(self, mesh: Mesh, num_copy: int = 1) -> None:
        super().__init__(mesh, num_copy)
        self.num_dof_per_dim = np.array((mesh.point.shape[0],), dtype=np.int64)
        self.num_dof = mesh.point.shape[0]
        # build cell dofs
        self.cell_dof[2] = self.mesh.cell[2]
        # build the facet dofs
        self.cell_dof[1] = self.mesh.cell[1]

    @staticmethod
    def _eval_basis(basis_id: int, qpts: np.ndarray) -> np.ndarray: # rdim(=1) * num_quad
        x = qpts[0]
        y = qpts[1]
        if basis_id == 0:
            basis = 1.0 - x - y
        elif basis_id == 1:
            basis = x
        elif basis_id == 2:
            basis = y
        return basis.reshape(1, -1)
    
    @staticmethod
    def _eval_grad(basis_id:int, qpts: np.ndarray) -> np.ndarray: # rdim(=1) * tdim(=2) * num_quad
        x = qpts[0]
        y = qpts[1]
        if basis_id == 0:
            data = np.vstack((-np.ones_like(x), -np.ones_like(y)))
        elif basis_id == 1:
            data = np.vstack((np.ones_like(x), np.zeros_like(y)))
        elif basis_id == 2:
            data = np.vstack((np.zeros_like(x), np.ones_like(y)))
        return data[np.newaxis, :, :]

class TriP2(TriElement):

    rdim: int = 1
    degree: int = 2
    num_dof_per_elem: int = 6
    trace_type = [NodeElement, LineP2]

    def __init__(self, mesh: Mesh, num_copy: int = 1) -> None:
        super().__init__(mesh, num_copy)
        Np = mesh.point.shape[0]
        edge_table = mesh.get_entities(1)
        self.num_dof_per_dim = np.array((Np, edge_table.nnz), dtype=np.int64)
        self.num_dof = np.sum(self.num_dof_per_dim).item()
        # build cell dofs
        self.cell_dof[2] = np.zeros((mesh.cell[2].shape[0], self.num_dof_per_elem+1), dtype=np.uint32)
        self.cell_dof[2][:, :3] = mesh.cell[2][:, :3]
        tri_edges = mesh.cell[2][:, [0,1,1,2,2,0]].reshape(-1, 3, 2)
        tri_edges = np.stack((np.min(tri_edges, axis=2), np.max(tri_edges, axis=2)), axis=2).reshape(-1, 2)
        idx = edge_table[tri_edges[:,0], tri_edges[:,1]]
        self.cell_dof[2][:, 3:-1] = idx.reshape(-1, 3) + Np - 1
        self.cell_dof[2][:, -1] = mesh.cell[2][:, -1]
        # build the facet dofs
        self.cell_dof[1] = np.zeros((mesh.cell[1].shape[0], self.trace_type[1].num_dof_per_elem + 1), dtype=np.uint32)
        self.cell_dof[1][:, :2] = mesh.cell[1][:, :2]
        edges = np.stack((np.min(mesh.cell[1][:, :-1], axis=1), np.max(mesh.cell[1][:, :-1], axis=1)), axis=1)
        self.cell_dof[1][:, 2] = edge_table[edges[:,0], edges[:,1]] + Np - 1
        self.cell_dof[1][:, -1] = mesh.cell[1][:, -1]
        # find also the dof locations
        self.dofloc = np.zeros((self.num_dof, mesh.gdim))
        self.dofloc[:Np, :] = mesh.point
        row_idx, col_idx = edge_table.nonzero()
        self.dofloc[Np:, :] = 0.5 * (mesh.point[row_idx, :] + mesh.point[col_idx, :])

    @staticmethod
    def _eval_basis(basis_id: int, qpts: np.ndarray) -> np.ndarray: # rdim(=1) * num_quad
        assert(basis_id < TriP2.num_dof_per_elem)
        x = qpts[0]
        y = qpts[1]
        if basis_id == 0:
            basis = 2.0*x**2 - 3.0*x + 1.0 + 2.0*y**2 - 3.0*y + 4.0*x*y
        elif basis_id == 1:
            basis = 2.0*x*(x-1.0/2)
        elif basis_id == 2:
            basis = 2.0*y*(y-1.0/2)
        elif basis_id == 3:
            basis = -4.0*x*(x+y-1)
        elif basis_id == 4:
            basis = 4.0*x*y
        elif basis_id == 5:
            basis = -4.0*y*(x+y-1)
        return basis.reshape(1, -1)
    
    @staticmethod
    def _eval_grad(basis_id:int, qpts: np.ndarray) -> np.ndarray: # rdim(=1) * tdim(=2) * num_quad
        assert(basis_id < TriP2.num_dof_per_elem)
        x = qpts[0]
        y = qpts[1]
        if basis_id == 0:
            data = np.vstack((4.0*x+4.0*y-3.0, 4.0*x+4.0*y-3.0))
        elif basis_id == 1:
            data = np.vstack((4.0*x-1.0, 0.0*y))
        elif basis_id == 2:
            data = np.vstack((0.0*x, 4.0*y-1.0))
        elif basis_id == 3:
            data = np.vstack((-8.0*x-4.0*y+4.0, -4.0*x))
        elif basis_id == 4:
            data = np.vstack((4.0*y, 4.0*x))
        elif basis_id == 5:
            data = np.vstack((-4.0*y, -4.0*x-8.0*y+4.0))
        return data[np.newaxis, :, :]
        
