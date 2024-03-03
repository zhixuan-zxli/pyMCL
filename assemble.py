from typing import Optional
import numpy as np
from scipy.sparse import csr_matrix
from mesh import *
from fe import *
from quadrature import *


class Measure:
    def __init__(self, mesh: Mesh, tdim: int, sub_id: Optional[int] = None) -> None:
        self.mesh = mesh
        self.tdim = tdim
        self.sub_id = sub_id

class QuadData(np.ndarray):
    """
    Array of size Nelem * Nquad, for assembly. 
    Attributes: grad
    """

    def __new__(cls, fe: FiniteElement, mea: Measure, quadOrder: int):
        dof = fe.getDof(mea.tdim, mea.sub_id)
        # get quad table
        obj = np.zeros().view(cls)
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
    fe: FiniteElement
    
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

class assembler:

    def __init__(self, 
                 test_space: FiniteElement, 
                 trial_space: Optional[FiniteElement], 
                 quadOrder: int) -> None:
        self.test_space = test_space
        self.trial_space = trial_space
        self.quadOrder = quadOrder
        if trial_space is not None:
            assert(test_space.mesh is trial_space.mesh)

    def assembleLinear(self, form, mea: Measure, **kwargs) -> np.ndarray:
        assert(mea.tdim <= self.mesh.tdim)
        
    # def _assembleLinearD2(self, form, dof, dof_flag, order: int, **kwargs) -> np.ndarray:
    #     pass

    # def _assembleLinearD1(self, form, mea: Measure, order: int, **kwargs) -> np.ndarray:
    #     pass
    
    def assembleBilinear(self, form, mea: Measure, **kwargs) -> csr_matrix:
        raise NotImplementedError
    