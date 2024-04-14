import numpy as np
from .refdom import RefCell, RefNode, RefLine, RefTri

class Element:

    ref_cell: RefCell
    tdim: int
    rdim: int
    degree: int
    dof_name: tuple[tuple[str]] 
    # of length tdim+1; the t-th tuple is of length num_dof_loc[t], with dof names therein
    dof_loc: tuple[np.ndarray] 
    # of length tdim+1; the t-th array is (num_dof_loc[t], t+1)
    num_local_dof: int
    
    @staticmethod
    def _eval(basis_id: int, qpts: np.ndarray) -> tuple[np.ndarray]: # (rdim, Nq), (rdim, tdim, Nq)
        raise NotImplementedError # will be implemented by subclasses
    
class NodeElement(Element):

    ref_cell: RefCell = RefNode()
    tdim: int = 0
    rdim: int = 1
    degree: int = 0
    dof_name: tuple[tuple[str]] = (
        ('u', ), # node
    )
    dof_loc: tuple[np.ndarray] = (
        np.array(((1.0,),)), # node
    )
    num_local_dof: int = 1

    @staticmethod
    def _eval(basis_id: int, qpts: np.ndarray) -> tuple[np.ndarray]:
        assert(basis_id == 0)
        return np.ones((1, qpts.shape[1])), \
            np.zeros((1, 0, qpts.shape[1]))


# =====================================================================
# Line elements    

class LineElement(Element):
    
    ref_cell: RefCell = RefLine()
    tdim: int = 1

class LineDG0(LineElement):

    rdim: int = 1
    degree: int = 0
    dof_name: tuple[tuple[str]] = (
        None, # node
        ('u', ), # edge
    )
    dof_loc: tuple[np.ndarray] = (
        None, # node, 
        np.array(((1.0/2, 1.0/2), )) # edge
    )
    num_local_dof: int = 1
    
    @staticmethod
    def _eval(basis_id: int, qpts: np.ndarray) -> tuple[np.ndarray]: 
        raise NotImplementedError
    

class LineP1(LineElement):

    rdim: int = 1
    degree: int = 1
    dof_name: tuple[tuple[str]] = (
        ('u', ), # node
        None, # edge
    )
    dof_loc: tuple[np.ndarray] = (
        np.array(((1.0,),)), # node
        None # edge
    )
    num_local_dof: int = 2

    @staticmethod
    def _eval(basis_id: int, qpts: np.ndarray) -> tuple[np.ndarray]: 
        x = qpts[0]
        if basis_id == 0:
            basis = 1.0 - x
            grad = -np.ones_like(x)
        elif basis_id == 1:
            basis = x
            grad = np.ones_like(x)
        return basis.reshape(1, -1), grad[np.newaxis, np.newaxis, :]

class LineP2(LineElement):
    
    rdim: int = 1
    degree: int = 2
    dof_name: tuple[tuple[str]] = (
        ('u', ), # node
        ('u', ), # edge
    )
    dof_loc: tuple[np.ndarray] = (
        np.array((1.0,)), # node
        np.array(((1.0/2, 1.0/2), )) # edge
    )
    num_local_dof: int = 3

    @staticmethod
    def _eval(basis_id: int, qpts: np.ndarray) -> tuple[np.ndarray]: 
        x = qpts[0]
        if basis_id == 0:
            basis = 2 * (x-0.5) * (x-1.0)
            grad = 4.0*x - 3.0
        elif basis_id == 1:
            basis = 2 * (x-0.5) * x
            grad = 4.0*x - 1.0
        elif basis_id == 2:
            basis = -4.0 * x * (x-1.0)
            grad = 4.0 - 8.0*x
        return basis[np.newaxis, :], grad[np.newaxis, np.newaxis, :]

# =====================================================================
# Triangular elements

class TriElement(Element):
    
    ref_cell: RefCell = RefTri()
    tdim: int = 2

class TriDG0(TriElement):

    rdim: int = 1
    degree: int = 0
    dof_name: tuple[tuple[str]] = (
        None, # node
        None, # edge
        ('u', ), # tri
    )
    dof_loc: tuple[np.ndarray] = (
        None, # node
        None, # edge
        np.array(((1.0/3, 1.0/3, 1.0/3),)) # tri
    )
    num_local_dof: int = 1

    @staticmethod
    def _eval(basis_id: int, qpts: np.ndarray) -> tuple[np.ndarray]:
        assert basis_id == 0
        return np.ones((1, qpts.shape[1])), np.zeros((1, 2, qpts.shape[1]))

class TriP1(TriElement):

    rdim: int = 1
    degree: int = 1
    dof_name: tuple[tuple[str]] = (
        ('u', ), # node
        None, # edge
        None, # tri
    )
    dof_loc: tuple[np.ndarray] = (
        np.array(((1.0,),)), # node
        None, # edge
        None # tri
    )
    num_local_dof: int = 3

    @staticmethod
    def _eval(basis_id: int, qpts: np.ndarray) -> tuple[np.ndarray]:
        x = qpts[0]
        y = qpts[1]
        if basis_id == 0:
            basis = 1.0 - x - y
            grad = np.vstack((-np.ones_like(x), -np.ones_like(y)))
        elif basis_id == 1:
            basis = x
            grad = np.vstack((np.ones_like(x), np.zeros_like(y)))
        elif basis_id == 2:
            basis = y
            grad = np.vstack((np.zeros_like(x), np.ones_like(y)))
        return basis[np.newaxis,:], grad[np.newaxis, :, :]
    
class TriDG1(TriElement):

    rdim: int = 1
    degree: int = 1
    dof_name: tuple[tuple[str]] = (
        None, # node
        None, # edge
        ('u', ), # tri
    )
    dof_loc: tuple[np.ndarray] = (
        None, # node
        None, # edge
        np.array((
            (0., 0., 1.0),
            (1., 0., 0.), 
            (0., 1., 0.)
        )) # tri
    )
    num_local_dof: int = 3

    @staticmethod
    def _eval(basis_id: int, qpts: np.ndarray) -> tuple[np.ndarray]:
        return TriP1._eval(basis_id, qpts)

class TriP2(TriElement):

    rdim: int = 1
    degree: int = 2
    dof_name: tuple[tuple[str]] = (
        ('u',), # node
        ('u',), # edge
        None
    )
    dof_loc: tuple[np.ndarray] = (
        np.array(((1.0,),)), # node
        np.array(((1.0/2, 1.0/2), )), # edge
        None # tri
    )
    num_local_dof: int = 6

    @staticmethod
    def _eval(basis_id: int, qpts: np.ndarray) -> tuple[np.ndarray]:
        assert basis_id < TriP2.num_local_dof
        x = qpts[0]
        y = qpts[1]
        if basis_id == 0:
            basis = 2.0*x**2 - 3.0*x + 1.0 + 2.0*y**2 - 3.0*y + 4.0*x*y
            grad = np.vstack((4.0*x+4.0*y-3.0, 4.0*x+4.0*y-3.0))
        elif basis_id == 1:
            basis = 2.0*x*(x-1.0/2)
            grad = np.vstack((4.0*x-1.0, 0.0*y))
        elif basis_id == 2:
            basis = 2.0*y*(y-1.0/2)
            grad = np.vstack((0.0*x, 4.0*y-1.0))
        elif basis_id == 3:
            basis = -4.0*x*(x+y-1.0)
            grad = np.vstack((-8.0*x-4.0*y+4.0, -4.0*x))
        elif basis_id == 4:
            basis = 4.0*x*y
            grad = np.vstack((4.0*y, 4.0*x))
        elif basis_id == 5:
            basis = -4.0*y*(x+y-1.0)
            grad = np.vstack((-4.0*y, -4.0*x-8.0*y+4.0))
        return basis[np.newaxis,:], grad[np.newaxis, :, :]
    
class TriDG2(TriElement):

    rdim: int = 1
    degree: int = 2
    dof_name: tuple[tuple[str]] = (
        None, # node
        None, # edge
        ('u', ), # tri
    )
    dof_loc: tuple[np.ndarray] = (
        None, # node
        None, # edge
        np.array((
            (0., 0., 1.0),
            (1., 0., 0.), 
            (0., 1., 0.), 
            (1.0/2, 0., 1.0/2), 
            (1.0/2, 1.0/2, 0.), 
            (0., 1.0/2, 1.0/2), 
        )) # tri
    )
    num_local_dof: int = 6

    @staticmethod
    def _eval(basis_id: int, qpts: np.ndarray) -> tuple[np.ndarray]:
        return TriP2._eval(basis_id, qpts)

    
# ====================================================
# derived element

class VectorElement(Element):
    
    base_elem: Element

    def __init__(self, base_elem: Element, num_copy: int) -> None:
        assert base_elem.rdim == 1, "Cannot build vector element from vector element. "
        self.base_elem = base_elem
        # modify the properties accordingly
        self.ref_cell = base_elem.ref_cell
        self.tdim = base_elem.tdim
        self.rdim = num_copy
        self.degree = base_elem.degree
        def _repeat(names: tuple[str]) -> tuple[str]:
            return sum((tuple(n + "_" + str(d) for d in range(num_copy)) for n in names), tuple())
        self.dof_name = tuple(_repeat(names) if names is not None else None for names in base_elem.dof_name)
        self.dof_loc = base_elem.dof_loc
        self.num_local_dof = base_elem.num_local_dof * num_copy
    
    def _eval(self, basis_id: int, qpts: np.ndarray) -> tuple[np.ndarray]:
        r = np.zeros((self.rdim, qpts.shape[1]))
        g = np.zeros((self.rdim, self.tdim, qpts.shape[1]))
        tr, tg = self.base_elem._eval(basis_id // self.rdim, qpts)
        r[basis_id % self.rdim], g[basis_id % self.rdim] = tr[0], tg[0]
        return r, g
        
