from typing import Optional
import numpy as np
from scipy.sparse import csr_matrix
from fe import Measure
from mesh import Mesh
from function import FiniteElement, QuadData, Function
from quadrature import Quadrature

    
class assembler:

    def __init__(self, 
                 test_space: FiniteElement, 
                 trial_space: Optional[FiniteElement], 
                 mea: Measure, 
                 quadOrder: int, 
                 geom_hint = None) -> None:
        self.test_space = test_space
        self.trial_space = trial_space
        self.mea = mea
        # test space
        self.test_dof = test_space.getCellDof(mea)
        self.test_basis = type(test_space) if mea.tdim == test_space.tdim else test_space.trace_type[mea.tdim]
        self.quadTable = Quadrature.getTable(test_space.ref_cell, quadOrder)
        # trial space
        self.trial_basis, self.trial_dof = self._get_basis_and_dof(trial_space)
        # geometric mapping space
        geom_space = test_space.mesh.coord_fe
        geom_basis, geom_dof = self._get_basis_and_dof(geom_space)
        if geom_hint is None:
            geom_hint = ("f", "grad", "dx", "inv_grad")
        self.geom_data = test_space.mesh.coord_map._get_quad_data(geom_basis, geom_dof, None, self.quadTable, geom_hint)

    def _get_basis_and_dof(self, fe: FiniteElement) -> tuple[type, np.ndarray]:
        """
        A helper to get the basis type and the DOF on the provided measure, given a finite element space. 
        """
        if fe is not None:
            if fe is self.test_space:
                return self.test_basis, self.test_dof
            else:
                basis_type = type(fe) if self.mea.tdim == fe.tdim else fe.trace_type[self.mea.tdim]
                return basis_type, fe.getCellDof(self.mea)
        else:
            return None, None

    def functional(self, form, **extra_args) -> float | np.ndarray:
        """
        form(x, w) : x the coordinates, w the extra functions
        """
        # convert the args into QuadData
        extra_data = dict()
        for k, v in extra_args.items():
            if isinstance(v, QuadData):
                extra_data[k] = v
            elif isinstance(v, Function):
                basis, dof = self._get_basis_and_dof(v.fe)
                extra_data[k] = v._get_quad_data(dof, self.mea.tdim, basis, self.quadTable, form.hint)
            else:
                raise RuntimeError("Unable to convert extra argument to QuadData. ")
        data = self.test_space.ref_cell.dx * form(self.geom_data, extra_data) # (rdim, Ne, num_quad)
        Ne = data.shape[1] if data.ndim > 2 else data.shape[0]
        Nq = data.shape[-1]
        data = data.reshape(-1, Nq) @ self.quadTable[-1, :]
        data = data.reshape(-1, Ne).sum(axis=1) # sum over all elements
        return data if data.size > 1 else data.item()


    def linear(self, form, **kwargs) -> np.ndarray:
        # assert(mea.tdim <= self.mesh.tdim)
        # basis_type = self.test_space
        # for _ in range(self.mesh.tdim - mea.tdim):
        #     basis_type = basis_type.trace_type
        pass
    
    def bilinear(self, form, **kwargs) -> csr_matrix:
        raise NotImplementedError
    
def setMeshMapping(mesh: Mesh, mapping: Optional[Function] = None):
    if mapping is None:
        # set an affine mapping
        if mesh.tdim == 2:
            from fe import TriP1
            mesh.coord_fe = TriP1(mesh, mesh.gdim)
            mesh.coord_map = Function(mesh.coord_fe)
            np.copyto(mesh.coord_map, mesh.point.T)
        else:
            raise NotImplementedError
    else:
        mesh.mapping = mapping
    