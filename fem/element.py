import numpy as np
from .refdom import RefNode, RefLine, RefTri

class Element:

    ref_cell: type
    tdim: int
    rdim: int
    degree: int
    num_dof_per_ent: tuple[int] # of length tdim+1    
    
    @staticmethod
    def _eval_basis(basis_id: int, qpts: np.ndarray) -> np.ndarray: # (rdim, Nq)
        raise NotImplementedError # will be implemented by subclasses
    
    @staticmethod
    def _eval_grad(basis_id: int, qpts: np.ndarray) -> np.ndarray:  # (rdim, tdim, Nq)
        raise NotImplementedError # will be implemented by subclasses
    
class NodeElement(Element):

    ref_cell: type = RefNode
    tdim: int = 0
    rdim: int = 1
    degree: int = 0
    num_dof_per_ent: tuple[int] = (1,)

    @staticmethod
    def _eval_basis(basis_id: int, qpts: np.ndarray) -> np.ndarray:
        assert(basis_id == 0)
        return np.ones((1, qpts.shape[1]))
    
    @staticmethod
    def _eval_grad(basis_id: int, qpts: np.ndarray) -> np.ndarray: 
        assert(basis_id == 0)
        return np.zeros((1, 0, qpts.shape[1]))


# =====================================================================
# Line elements    

class LineElement(Element):
    
    ref_cell: type = RefLine
    tdim: int = 1

class LineDG0(LineElement):

    rdim: int = 1
    degree: int = 0
    num_dof_per_ent: tuple[int] = (0, 1)
    
    @staticmethod
    def _eval_basis(basis_id: int, qpts: np.ndarray) -> np.ndarray: 
        raise NotImplementedError
    
    @staticmethod
    def _eval_grad(basis_id: int, qpts: np.ndarray) -> np.ndarray: 
        raise NotImplementedError

class LineP1(LineElement):

    rdim: int = 1
    degree: int = 1
    num_dof_per_ent: tuple[int] = (1, 0)

    @staticmethod
    def _eval_basis(basis_id: int, qpts: np.ndarray) -> np.ndarray: 
        x = qpts[0]
        if basis_id == 0:
            basis = 1.0 - x
        elif basis_id == 1:
            basis = x
        return basis.reshape(1, -1)
    
    @staticmethod
    def _eval_grad(basis_id: int, qpts: np.ndarray) -> np.ndarray: 
        x = qpts[0]
        if basis_id == 0:
            data = -np.ones_like(x)
        elif basis_id == 1:
            data = np.ones_like(x)
        return data[np.newaxis, np.newaxis, :]

class LineP2(LineElement):
    
    rdim: int = 1
    degree: int = 2
    num_dof_per_ent: tuple[int] = (1, 1)

    @staticmethod
    def _eval_basis(basis_id: int, qpts: np.ndarray) -> np.ndarray: 
        x = qpts[0]
        if basis_id == 0:
            basis = 2 * (x-0.5) * (x-1.0)
        elif basis_id == 1:
            basis = 2 * (x-0.5) * x
        elif basis_id == 2:
            basis = -4.0 * x * (x-1.0)
        return basis[np.newaxis, :]
    
    @staticmethod
    def _eval_grad(basis_id: int, qpts: np.ndarray) -> np.ndarray: 
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

class TriElement(Element):
    
    ref_cell: type = RefTri
    tdim: int = 2

class TriDG0(TriElement):

    rdim: int = 1
    degree: int = 0
    num_dof_per_ent: tuple[int] = (0, 0, 1)

    @staticmethod
    def _eval_basis(basis_id: int, qpts: np.ndarray) -> np.ndarray:
        assert basis_id == 0
        return np.ones((1, qpts.shape[1]))
    
    @staticmethod
    def _eval_grad(basis_id: int, qpts: np.ndarray) -> np.ndarray:
        return np.zeros((1, 2, qpts.shape[1]))

class TriP1(TriElement):

    rdim: int = 1
    degree: int = 1
    num_dof_per_ent: tuple[int] = (1, 0, 0)

    @staticmethod
    def _eval_basis(basis_id: int, qpts: np.ndarray) -> np.ndarray:
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
    def _eval_grad(basis_id:int, qpts: np.ndarray) -> np.ndarray:
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
    num_dof_per_ent: tuple[int] = (1,1,0)

    @staticmethod
    def _eval_basis(basis_id: int, qpts: np.ndarray) -> np.ndarray:
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
    def _eval_grad(basis_id:int, qpts: np.ndarray) -> np.ndarray:
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
    
# ====================================================
# derived element

class VectorElement(Element):
    
    ref_cell: type
    tdim: int
    rdim: int
    degree: int
    num_dof_per_ent: tuple[int] # of length tdim+1

    def __init__(self, base_elem: Element) -> None:
        assert base_elem.rdim == 1
        raise NotImplementedError
