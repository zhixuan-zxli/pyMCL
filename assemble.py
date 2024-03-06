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

    # A helper to get the basis type and the DOF on the provided measure, given a finite element space.
    def _get_basis_and_dof(self, fe: FiniteElement):
        if fe is not None:
            if fe is self.test_space:
                return self.test_basis, self.test_dof
            else:
                basis_type = type(fe) if self.mea.tdim == fe.tdim else fe.trace_type[self.mea.tdim]
                return basis_type, fe.getCellDof(self.mea)
        else:
            return None, None
        
    def _get_basis_quad_data(self, basis_type: type, basis_id: int, rdim: int, hint) -> QuadData:
        data = None
        if "f" in hint:
            data = np.zeros((rdim, 1, self.quadTable.shape[1])) # (rdim, 1, num_quad)
            data[:,0,:] = basis_type._eval_basis(basis_id, self.quadTable) 
        data = QuadData(data)
        if "grad" in hint:
            basis_grad = basis_type._eval_grad(basis_id, self.quadTable) # (rdim, tdim, num_quad)
            data.grad = np.einsum("ij...,jk...->ik...", \
                                  basis_grad[:,:,np.newaxis,:], self.geom_data.inv_grad) # (rdim, gdim, Ne, num_quad)
        return data
    
    def _transform_extra_args(self, hint, **extra_args):
        # convert the args into QuadData
        extra_data = dict()
        for k, v in extra_args.items():
            if isinstance(v, QuadData):
                extra_data[k] = v
            elif isinstance(v, Function):
                basis, dof = self._get_basis_and_dof(v.fe)
                extra_data[k] = v._get_quad_data(dof, self.mea.tdim, basis, self.quadTable, hint)
            else:
                raise RuntimeError("Unable to convert the argument to QuadData. ")
        return extra_data

    def functional(self, form, **extra_args) -> float | np.ndarray:
        """
        form(x, w) : x the coordinates, w the extra functions
        """
        extra_data = self._transform_extra_args(form.hint, **extra_args)
        Ne, Nq = self.test_dof.shape[0], self.quadTable.shape[1]
        data = self.test_space.ref_cell.dx * form(self.geom_data, extra_data) # (rdim, Ne, num_quad)
        assert data.shape[1:] == (Ne, Nq)
        data = data.reshape(-1, Nq) @ self.quadTable[-1, :]
        data = data.reshape(-1, Ne).sum(axis=1) # sum over all elements
        return data if data.size > 1 else data.item()


    def linear(self, form, **extra_args) -> np.ndarray:
        """
        form(psi, x, w) : psi the test function, x the coordinates, w the extra functions
        """
        extra_data = self._transform_extra_args(form.hint, **extra_args)
        # do quadrature
        num_copy = self.test_space.num_copy
        rdim = max(self.test_space.rdim, num_copy)
        num_dof_per_elem = self.test_basis.num_dof_per_elem
        Ne = self.test_dof.shape[0]
        Nq = self.quadTable.shape[1]
        values = np.zeros((rdim, num_dof_per_elem, Ne), dtype=np.float64)
        for i in range(num_dof_per_elem):
            psi = self._get_basis_quad_data(self.test_basis, i, rdim, form.hint) # (rdim, 1, Nq)
            form_data = form(psi, self.geom_data, extra_data) # (rdim, Ne, Nq)
            assert form_data.shape == (rdim, Ne, Nq)
            values[:, i, :] = \
                (form_data.reshape(-1, Nq) @ self.quadTable[-1, :] * self.test_basis.ref_cell.dx).reshape(rdim, Ne)

        indices = np.zeros((num_copy, num_dof_per_elem, Ne), dtype=np.uint32) # test here
        indices[:, :, :] = self.test_dof.T
        if num_copy > 1:
            indices *= num_copy
            for c in range(num_copy):
                indices[c,:,:] += c
        
        vec = np.bincount(indices.ravel(), values.ravel(), minlength=self.test_space.num_dof * rdim)
        return vec

    
    def bilinear(self, form, **extra_args) -> csr_matrix:
        """
        form(phi, psi, x, w) : 
        phi the trial function, psi the test function, x the coordinates, w the extra functions
        """
        assert self.trial_space is not None
        extra_data = self._transform_extra_args(form.hint, **extra_args)
        num_copy = self.test_space.num_copy, self.trial_space.num_copy
        rdim = max(self.test_space.rdim, num_copy[0]), \
          max(self.trial_space.rdim, num_copy[1])
        Ne = self.test_dof.shape[0]
        assert Ne == self.trial_dof.shape[0]
        Nq = self.quadTable.shape[1]
        values = np.zeros(
            (num_copy[0], num_copy[1], 
             self.test_basis.num_dof_per_elem, self.trial_basis.num_dof_per_elem, Ne)
             )
        row_idx = np.zeros_like(values, dtype=np.uint32)
        col_idx = np.zeros_like(values, dtype=np.uint32)
        for i in range(self.test_basis.num_dof_per_elem):
            psi = self._get_basis_quad_data(self.test_basis, i, rdim[0], form.hint) # (rdim[0], 1, Nq)
            for j in range(self.trial_basis.num_dof_per_elem):
                phi = self._get_basis_quad_data(self.test_basis, j, rdim[1], form.hint) # (rdim[1], 1, Nq)
                form_data = form(phi, psi, self.geom_data, extra_data) # (rdim[0], rdim[1], Ne, Nq)
                row_idx[:,:,i,j,:] = self.test_dof[:,i]
                col_idx[:,:,i,j,:] = self.trial_dof[:,j]
                # if num_copy > 0 ...
    



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
    