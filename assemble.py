from typing import Optional
import numpy as np
from scipy.sparse import csr_matrix
from fe import Measure
from mesh import Mesh
from function import FiniteElement, Function
from quadrature import Quadrature

    

class assembler:

    def __init__(self, 
                 test_space: FiniteElement, 
                 trial_space: Optional[FiniteElement], 
                 mea: Measure, 
                 quadOrder: int) -> None:
        self.test_space = test_space
        self.trial_space = trial_space
        self.quadOrder = quadOrder
        self.cell_dof = test_space.getCellDof(mea.tdim, mea.sub_id)

    def assembleFunctional(self, form, **kwargs) -> float | np.ndarray:
        return 0.0

    def assembleLinear(self, form, **kwargs) -> np.ndarray:
        # assert(mea.tdim <= self.mesh.tdim)
        # basis_type = self.test_space
        # for _ in range(self.mesh.tdim - mea.tdim):
        #     basis_type = basis_type.trace_type
        pass
    
    def assembleBilinear(self, form, **kwargs) -> csr_matrix:
        raise NotImplementedError
    
def setMeshMapping(mesh: Mesh, mapping: Optional[Function] = None):
    if mapping is None:
        # set an affine mapping
        raise NotImplementedError
    else:
        mesh.mapping = mapping
    