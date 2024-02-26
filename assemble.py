from typing import Optional, Type
import numpy as np
from scipy.sparse import csr_matrix
from mesh import Mesh
from fem import FiniteElement
from quadrature import QuadTable

class QuadData(np.ndarray):
    """
    Array of size Nelem * Nquad, for assembly. 
    Attributes: grad
    """
    def __new__(cls, fe: FiniteElement, num_quad: int):
        obj = np.zeros((fe.cell_dof.shape[0], num_quad))
        obj.fe = fe
        obj.grad = None
        return obj
    
    def __array_finalize__(self, obj) -> None:
        if obj is None: return
        self.fe = getattr(obj, "fe", None)
        self.grad = getattr(obj, "grad", None)

    def __array_wrap__(self, out_arr, context=None):
        # invalidate the attributes
        return np.array(out_arr)

class Function(np.ndarray):
    """
    Array of size num_dof. 
    """
    def __new__(cls, fe: FiniteElement):
        obj = np.zeros((fe.num_dof, )).view(cls)
        obj.fe = fe
        return obj
    
    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self.fe = getattr(obj, "fe", None)

    def __array_wrap__(self, out_arr, context=None):
        out_arr.fe = self.fe
        return out_arr
    
    def _toQuadData(self) -> QuadData:
        pass

class Measure:
    def __init__(self, mesh: Mesh, type: str, sub_id: Optional[int] = None) -> None:
        self.mesh = mesh
        self.sub_id = sub_id

class bulkMeasure(Measure):
    def __init__(self, mesh: Mesh, type: str, sub_id: Optional[int] = None) -> None:
        super().__init__(mesh, type, sub_id)

    # routine for generating gradients, cell normal and Jacobian

class surfaceMeasure(Measure):
    def __init__(self, mesh: Mesh, type: str, sub_id: Optional[int] = None) -> None:
        super().__init__(mesh, type, sub_id)

    # routine for generating gradients, cell normal and Jacobian


class assembler:

    def __init__(self, 
                 test_space: FiniteElement, 
                 trial_space: Optional[FiniteElement], 
                 quadOrder: int) -> None:
        self.test_space = test_space
        self.trial_space = trial_space
        self.quadOrder = quadOrder
        assert(test_space is not None)
        if trial_space is not None:
            assert(test_space.mesh is trial_space.mesh)

    def assembleLinear(self, form, mea: Measure, **kwargs) -> np.ndarray:
        raise NotImplementedError
    
    def assembleBilinear(self, form, mea: Measure, **kwargs) -> csr_matrix:
        raise NotImplementedError
    