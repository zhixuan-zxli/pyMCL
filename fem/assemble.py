from typing import Optional
import numpy as np
from scipy.sparse import csr_array
from .mesh import Measure
from .function import FiniteElement, FieldData, Function
from .quadrature import Quadrature

class Form:
    expr: callable
    hint: tuple[str]

    def __init__(self, expr, *hint) -> None:
        self.expr = expr
        if len(hint) > 0:
            self.hint = hint
        else:
            self.hint = ("f", "grad")
    def __call__(self, *args, **kwds):
        return self.expr(*args, **kwds)
    
class assembler:

    def __init__(self, 
                 test_space: FiniteElement, 
                 trial_space: Optional[FiniteElement], 
                 mea: Measure, 
                 order: int, 
                 geom_hint = None) -> None:
        self.test_space = test_space
        self.trial_space = trial_space
        self.mea = mea
        # test space
        self.test_dof = test_space.getCellDof(mea)
        self.test_basis = type(test_space) if mea.tdim == test_space.tdim else test_space.trace_type[mea.tdim]
        self.quadTable = Quadrature.getTable(self.test_basis.ref_cell, order)
        # trial space
        self.trial_basis, self.trial_dof = self._get_basis_and_dof(trial_space)
        # geometric mapping space
        geom_space = test_space.mesh.coord_fe
        self.geom_basis, self.geom_dof = self._get_basis_and_dof(geom_space)
        if geom_hint is None:
            if mea.tdim == test_space.mesh.gdim:
                self.geom_hint = ("f", "grad", "dx", "inv_grad")
            elif mea.tdim == test_space.mesh.gdim-1: 
                self.geom_hint = ("f", "grad", "dx", "inv_grad", "n")
            else: # for codimension greater than one ...
                self.geom_hint = ("f", )
        else:
            self.geom_hint = geom_hint
        self.updateGeometry()

    def updateGeometry(self) -> None:
        coord_map = self.test_space.mesh.coord_map
        self.geom_data = coord_map._get_quad_data(self.geom_basis, self.geom_dof, None, self.quadTable, self.geom_hint)

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
        
    def _get_basis_quad_data(self, basis_type: type, basis_id: int, rdim: int, hint) -> FieldData:
        data = None
        if "f" in hint:
            data = np.zeros((rdim, 1, self.quadTable.shape[1])) # (rdim, 1, num_quad)
            data[:,0,:] = basis_type._eval_basis(basis_id, self.quadTable) 
        data = FieldData(data)
        if "grad" in hint:
            basis_grad = np.zeros((rdim, basis_type.tdim, self.quadTable.shape[1]))
            basis_grad[:,:,:] = basis_type._eval_grad(basis_id, self.quadTable) # (rdim, tdim, num_quad)
            data.grad = np.einsum("ij...,jk...->ik...", \
                                  basis_grad[:,:,np.newaxis,:], self.geom_data.inv_grad) # (rdim, gdim, Ne, num_quad)
        return data
    
    def _transform_extra_args(self, hint, **extra_args):
        # convert the args into QuadData
        extra_data = dict()
        for k, v in extra_args.items():
            if isinstance(v, Function):
                basis, dof = self._get_basis_and_dof(v.fe)
                extra_data[k] = v._get_quad_data(basis, dof, self.geom_data, self.quadTable, hint)
            else:
                extra_data[k] = v # for other data, leave it as it is
        return extra_data

    def functional(self, form: Form, **extra_args) -> float | np.ndarray:
        """
        form(x, w) : x the coordinates, w the extra functions
        """
        extra_data = self._transform_extra_args(form.hint, **extra_args)
        Ne, Nq = self.test_dof.shape[0], self.quadTable.shape[1]
        data = self.test_space.ref_cell.dx * form(self.geom_data, **extra_data) # (rdim, Ne, num_quad)
        assert data.shape[1:] == (Ne, Nq)
        data = data.reshape(-1, Nq) @ self.quadTable[-1, :]
        data = data.reshape(-1, Ne).sum(axis=1) # sum over all elements
        return data if data.size > 1 else data.item()


    def linear(self, form: Form, **extra_args) -> np.ndarray:
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
            form_data = form(psi, self.geom_data, **extra_data) # (rdim, Ne, Nq)
            assert form_data.shape == (rdim, Ne, Nq)
            values[:, i, :] = \
                (form_data.reshape(-1, Nq) @ self.quadTable[-1, :] * self.test_basis.ref_cell.dx).reshape(rdim, Ne)

        indices = np.zeros((num_copy, num_dof_per_elem, Ne), dtype=np.uint32) # test here
        indices[:, :, :] = self.test_dof.T
        if num_copy > 1:
            indices *= num_copy
            for c in range(num_copy):
                indices[c,:,:] += c
        
        vec = np.bincount(indices.reshape(-1), weights=values.reshape(-1), minlength=self.test_space.num_dof * num_copy)
        return vec.reshape(-1, 1)

    
    def bilinear(self, form: Form, **extra_args) -> csr_array:
        """
        form(psi, phi, x, w) : 
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
                phi = self._get_basis_quad_data(self.trial_basis, j, rdim[1], form.hint) # (rdim[1], 1, Nq)
                form_data = form(psi, phi, self.geom_data, **extra_data) # (rdim[0], rdim[1], Ne, Nq)
                values[:,:,i,j,:] = \
                  (form_data.reshape(-1, Nq) @ self.quadTable[-1,:] * self.test_basis.ref_cell.dx).reshape(rdim[0], rdim[1], Ne)
                row_idx[:,:,i,j,:] = self.test_dof[:,i]
                col_idx[:,:,i,j,:] = self.trial_dof[:,j]
                # fix the indices for the multi-component case
                if num_copy[0] > 1:
                    row_idx[:,:,i,j,:] *= num_copy[0]
                    for ci in range(num_copy[0]):
                        row_idx[ci,:,i,j,:] += ci
                if num_copy[1] > 1:
                    col_idx[:,:,i,j,:] *= num_copy[1]
                    for cj in range(num_copy[1]):
                        col_idx[:,cj,i,j,:] += cj
        #
        shape = (self.test_space.num_dof * num_copy[0], self.trial_space.num_dof * num_copy[1])
        mat = csr_array((values.reshape(-1), (row_idx.reshape(-1), col_idx.reshape(-1))), shape=shape)
        return mat
    