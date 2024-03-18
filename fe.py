import numpy as np
from mesh import Mesh, Measure
from scipy.sparse import csr_array

class RefCell:
    tdim: int
    dx: float

class RefNode(RefCell):
    tdim: int = 0
    dx: float = 1.0

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
    periodic: bool
    # cell_dof

    def __init__(self, mesh: Mesh, num_copy: int = 1, periodic: bool = False) -> None:
        self.mesh = mesh
        self.num_copy = num_copy
        self.periodic = periodic
        if periodic:
            assert hasattr(mesh, "constraint_table")

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
    
    def __init__(self, mesh: Mesh, num_copy: int = 1, periodic: bool = False) -> None:
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

    def __init__(self, mesh: Mesh, num_copy: int = 1, periodic: bool = False) -> None:
        super().__init__(mesh, num_copy, periodic)
        assert(mesh.cell[1].shape[0] > 0)
        self.cell_dof = [None] * 2

class LineDG0(LineElement):

    rdim: int = 1
    degree: int = 0
    num_dof_per_elem: int = 1
    trace_type = [NodeElement]

    def __init__(self, mesh: Mesh, num_copy: int = 1, periodic: bool = False) -> None:
        raise NotImplementedError
    
    @staticmethod
    def _eval_basis(basis_id: int, qpts: np.ndarray) -> np.ndarray: # (rdim, Nq)
        raise NotImplementedError
    
    @staticmethod
    def _eval_grad(basis_id: int, qpts: np.ndarray) -> np.ndarray: # (rdim, tdim, Nq)
        raise NotImplementedError

class LineP1(LineElement):

    rdim: int = 1
    degree: int = 1
    num_dof_per_elem: int = 2
    trace_type = [NodeElement]

    def __init__(self, mesh: Mesh, num_copy: int = 1, periodic: bool = False) -> None:
        super().__init__(mesh, num_copy, periodic)
        self.num_dof_per_dim = np.array((mesh.point.shape[0],), dtype=np.int64)
        self.num_dof = mesh.point.shape[0]
        # build cell dofs
        self.cell_dof[1] = mesh.cell[1].copy()
        # build also doflocs
        self.dofloc = mesh.point
        # apply remap for periodic
        if periodic:
            point_remap = mesh._get_point_remap()
            self.cell_dof[1][:,:-1] = point_remap[mesh.cell[1][:,:-1]]
            self.dof_remap = point_remap

    @staticmethod
    def _eval_basis(basis_id: int, qpts: np.ndarray) -> np.ndarray: # (rdim, Nq)
        x = qpts[0]
        if basis_id == 0:
            basis = 1.0 - x
        elif basis_id == 1:
            basis = x
        return basis.reshape(1, -1)
    
    @staticmethod
    def _eval_grad(basis_id: int, qpts: np.ndarray) -> np.ndarray: # (rdim, tdim, Nq)
        x = qpts[0]
        if basis_id == 0:
            data = -np.ones_like(x)
        elif basis_id == 1:
            data = np.ones_like(x)
        return data[np.newaxis, np.newaxis, :]

class LineP2(LineElement):
    
    rdim: int = 1
    degree: int = 2
    num_dof_per_elem: int = 3
    trace_type = [NodeElement]

    def __init__(self, mesh: Mesh, num_copy: int = 1, periodic: bool = False) -> None:
        super().__init__(mesh, num_copy, periodic)
        Np = mesh.point.shape[0]
        Ne = mesh.cell[1].shape[0]
        self.num_dof_per_dim = np.array((Np, Ne), dtype=np.int64)
        self.num_dof = Np + Ne
        # build cell dofs
        self.cell_dof[1] = np.zeros((Ne, 4), dtype=np.uint32)
        self.cell_dof[1][:,:2] = mesh.cell[1][:, :2]
        self.cell_dof[1][:,2] = np.arange(Ne) + Np
        self.cell_dof[1][:, -1] = mesh.cell[1][:, -1]
        # build also dof locations
        self.dofloc = np.zeros((self.num_dof, mesh.gdim))
        self.dofloc[:Np, :] = mesh.point
        self.dofloc[Np:, :] = 0.5 * (mesh.point[mesh.cell[1][:, 0], :] + mesh.point[mesh.cell[1][:, 1], :])
        # apply remap for periodic BC
        if periodic:
            point_remap = mesh._get_point_remap()
            self.cell_dof[1][:,:2] = point_remap[mesh.cell[1][:,:2]]
            self.dof_remap = np.arange(self.num_dof, dtype=np.uint32)
            self.dof_remap[:Np] = point_remap

    @staticmethod
    def _eval_basis(basis_id: int, qpts: np.ndarray) -> np.ndarray: # (rdim, Nq)
        x = qpts[0]
        if basis_id == 0:
            basis = 2 * (x-0.5) * (x-1.0)
        elif basis_id == 1:
            basis = 2 * (x-0.5) * x
        elif basis_id == 2:
            basis = -4.0 * x * (x-1.0)
        return basis[np.newaxis, :]
    
    @staticmethod
    def _eval_grad(basis_id: int, qpts: np.ndarray) -> np.ndarray: # (rdim, tdim, Nq)
        x = qpts[0]
        if basis_id == 0:
            data = 4.0*x - 3.0
        elif basis_id == 1:
            data = 4.0*x - 1.0
        elif basis_id == 2:
            data = 4.0 - 8.0*x
        return data[np.newaxis, np.newaxis, :]

# =====================================================================
# Triangular elements

class TriElement(FiniteElement):
    
    ref_cell = RefTri
    tdim: int = 2

    def __init__(self, mesh: Mesh, num_copy: int = 1, periodic: bool = False) -> None:
        super().__init__(mesh, num_copy, periodic)
        assert(mesh.cell[2].shape[0] > 0)
        self.cell_dof = [None] * 3

class TriDG0(TriElement):

    rdim: int = 1
    degree: int = 0
    num_dof_per_elem: int = 1
    trace_type = [NodeElement, LineDG0]

    def __init__(self, mesh: Mesh, num_copy: int = 1, periodic: bool = False) -> None:
        super().__init__(mesh, num_copy, periodic)
        Nt = mesh.cell[2].shape[0]
        self.num_dof_per_dim = np.array((0, 0, Nt), dtype=np.int64)
        self.num_dof = Nt
        # build cell dof
        self.cell_dof[2] = np.zeros((Nt, 2), dtype=np.uint32)
        self.cell_dof[2][:,0] = np.arange(Nt)
        self.cell_dof[2][:,1] = mesh.cell[2][:,-1]
        # build facet dof
        # todo
        #
        if periodic:
            pass

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

    def __init__(self, mesh: Mesh, num_copy: int = 1, periodic: bool = False) -> None:
        super().__init__(mesh, num_copy, periodic)
        self.num_dof_per_dim = np.array((mesh.point.shape[0],), dtype=np.int64)
        self.num_dof = mesh.point.shape[0]
        # build cell dofs
        self.cell_dof[2] = mesh.cell[2].copy()
        # build the facet dofs
        self.cell_dof[1] = mesh.cell[1].copy()
        # also set the doflocs
        self.dofloc = mesh.point
        # apply remap for periodic BC
        if periodic:
            point_remap = mesh._get_point_remap()
            self.cell_dof[2][:,:-1] = point_remap[mesh.cell[2][:,:-1]]
            self.cell_dof[1][:,:-1] = point_remap[mesh.cell[1][:,:-1]]
            self.dof_remap = point_remap

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

    edge_map: csr_array

    def __init__(self, mesh: Mesh, num_copy: int = 1, periodic: bool = False) -> None:
        super().__init__(mesh, num_copy, periodic)
        Np, Nt = mesh.point.shape[0], mesh.cell[2].shape[0]
        # build cell dofs
        self.cell_dof[2] = np.zeros((Nt, self.num_dof_per_elem+1), dtype=np.uint32)
        self.cell_dof[2][:, :3] = mesh.cell[2][:, :3]
        self.edge_map, edges = mesh._get_edges_from_tri(Np, mesh.cell[2])
        idx = self.edge_map[edges[:,0], edges[:,1]]
        self.cell_dof[2][:, 3:-1] = idx.reshape(-1, 3) + Np - 1
        self.cell_dof[2][:, -1] = mesh.cell[2][:, -1]
        # set the num dof now
        self.num_dof_per_dim = np.array((Np, self.edge_map.nnz), dtype=np.int64)
        self.num_dof = Np + self.edge_map.nnz
        assert self.edge_map.nnz == Np + Nt - 1
        # build the facet dofs
        self.cell_dof[1] = np.zeros((mesh.cell[1].shape[0], self.trace_type[1].num_dof_per_elem + 1), dtype=np.uint32)
        self.cell_dof[1][:, :2] = mesh.cell[1][:, :2]
        edges = np.stack((np.min(mesh.cell[1][:, :-1], axis=1), np.max(mesh.cell[1][:, :-1], axis=1)), axis=1)
        self.cell_dof[1][:, 2] = self.edge_map[edges[:,0], edges[:,1]] + Np - 1
        self.cell_dof[1][:, -1] = mesh.cell[1][:, -1]
        # find also the dof locations
        self.dofloc = np.zeros((self.num_dof, mesh.gdim))
        self.dofloc[:Np, :] = mesh.point
        row_idx, col_idx = self.edge_map.nonzero()
        self.dofloc[Np:, :] = 0.5 * (mesh.point[row_idx, :] + mesh.point[col_idx, :])
        # apply remap for periodic BC
        if periodic:
            point_remap = mesh._get_point_remap()
            edge_remap = np.arange(self.edge_map.nnz, dtype=np.uint32)
            for corr in mesh.constraint_table:
                point_map = np.arange(Np, dtype=np.uint32)
                point_map[corr[:,0]] = corr[:,1]
                row_remap = point_map[row_idx]
                col_remap = point_map[col_idx]
                idx = np.nonzero((row_remap != row_idx) & (col_remap != col_idx))[0]
                temp = self.edge_map[np.minimum(row_remap[idx], col_remap[idx]), np.maximum(row_remap[idx], col_remap[idx])]
                assert np.all(temp >= 1)
                edge_remap[idx] = temp - 1
            self.dof_remap = np.concatenate((point_remap, edge_remap + Np))
            self.cell_dof[2][:,:-1] = self.dof_remap[self.cell_dof[2][:,:-1]]
            self.cell_dof[1][:,:-1] = self.dof_remap[self.cell_dof[1][:,:-1]]

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
        
