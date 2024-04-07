from typing import Optional, Callable, Union
import numpy as np
from scipy.sparse import csr_array
from .function import FunctionSpace, Function, QuadData
from .funcbasis import FunctionBasis
from .measure import Measure
from .quadrature import Quadrature
from .refdom import ref_doms
    
class Form:

    form: Callable

    def __init__(self, 
                 form: Callable) -> None:
        self.form = form

    def functional(self, mea: Measure, **extra_args) -> Union[float, np.ndarray]:
        """
        Assemble a functional. 
        """
        dx_ref = ref_doms[mea.dim].dx
        quad_tab = mea.quad_tab
        Nq = mea.quad_tab.shape[1]
        data = dx_ref * self.form(mea.x, **extra_args) # (rdim, Ne, num_quad)
        rdim = data.shape[0]
        data = data.reshape(-1, Nq) @ quad_tab[-1, :]
        data = data.reshape(rdim, -1).sum(axis=1) # sum over all elements
        return data if data.size > 1 else data.item()


    # def linear(self, form: Form, **extra_args) -> np.ndarray:
    #     """
    #     form(psi, x, w) : psi the test function, x the coordinates, w the extra functions
    #     """
    #     extra_data = self._transform_extra_args(form.hint, **extra_args)
    #     # do quadrature
    #     num_copy = self.test_space.num_copy
    #     rdim = max(self.test_space.rdim, num_copy)
    #     num_dof_per_elem = self.test_basis.num_dof_per_elem
    #     Ne = self.test_dof.shape[0]
    #     Nq = self.quadTable.shape[1]
    #     values = np.zeros((rdim, num_dof_per_elem, Ne), dtype=np.float64)
    #     for i in range(num_dof_per_elem):
    #         psi = self._get_basis_quad_data(self.test_basis, i, rdim, form.hint) # (rdim, 1, Nq)
    #         form_data = form(psi, self.geom_data, **extra_data) # (rdim, Ne, Nq)
    #         assert form_data.shape == (rdim, Ne, Nq)
    #         values[:, i, :] = \
    #             (form_data.reshape(-1, Nq) @ self.quadTable[-1, :] * self.test_basis.ref_cell.dx).reshape(rdim, Ne)

    #     indices = np.zeros((num_copy, num_dof_per_elem, Ne), dtype=np.uint32) # test here
    #     indices[:, :, :] = self.test_dof.T
    #     if num_copy > 1:
    #         indices *= num_copy
    #         for c in range(num_copy):
    #             indices[c,:,:] += c
        
    #     vec = np.bincount(indices.reshape(-1), weights=values.reshape(-1), minlength=self.test_space.num_dof * num_copy)
    #     return vec.reshape(-1, 1)

    
    # def bilinear(self, form: Form, **extra_args) -> csr_array:
    #     """
    #     form(psi, phi, x, w) : 
    #     phi the trial function, psi the test function, x the coordinates, w the extra functions
    #     """
    #     assert self.trial_space is not None
    #     extra_data = self._transform_extra_args(form.hint, **extra_args)
    #     num_copy = self.test_space.num_copy, self.trial_space.num_copy
    #     rdim = max(self.test_space.rdim, num_copy[0]), \
    #       max(self.trial_space.rdim, num_copy[1])
    #     Ne = self.test_dof.shape[0]
    #     assert Ne == self.trial_dof.shape[0]
    #     Nq = self.quadTable.shape[1]
    #     values = np.zeros(
    #         (num_copy[0], num_copy[1], 
    #          self.test_basis.num_dof_per_elem, self.trial_basis.num_dof_per_elem, Ne)
    #          )
    #     row_idx = np.zeros_like(values, dtype=np.uint32)
    #     col_idx = np.zeros_like(values, dtype=np.uint32)
    #     for i in range(self.test_basis.num_dof_per_elem):
    #         psi = self._get_basis_quad_data(self.test_basis, i, rdim[0], form.hint) # (rdim[0], 1, Nq)
    #         for j in range(self.trial_basis.num_dof_per_elem):
    #             phi = self._get_basis_quad_data(self.trial_basis, j, rdim[1], form.hint) # (rdim[1], 1, Nq)
    #             form_data = form(psi, phi, self.geom_data, **extra_data) # (rdim[0], rdim[1], Ne, Nq)
    #             values[:,:,i,j,:] = \
    #               (form_data.reshape(-1, Nq) @ self.quadTable[-1,:] * self.test_basis.ref_cell.dx).reshape(rdim[0], rdim[1], Ne)
    #             row_idx[:,:,i,j,:] = self.test_dof[:,i]
    #             col_idx[:,:,i,j,:] = self.trial_dof[:,j]
    #             # fix the indices for the multi-component case
    #             if num_copy[0] > 1:
    #                 row_idx[:,:,i,j,:] *= num_copy[0]
    #                 for ci in range(num_copy[0]):
    #                     row_idx[ci,:,i,j,:] += ci
    #             if num_copy[1] > 1:
    #                 col_idx[:,:,i,j,:] *= num_copy[1]
    #                 for cj in range(num_copy[1]):
    #                     col_idx[:,cj,i,j,:] += cj
    #     #
    #     shape = (self.test_space.num_dof * num_copy[0], self.trial_space.num_dof * num_copy[1])
    #     mat = csr_array((values.reshape(-1), (row_idx.reshape(-1), col_idx.reshape(-1))), shape=shape)
    #     return mat
    