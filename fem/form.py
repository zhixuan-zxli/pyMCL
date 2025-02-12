from typing import Callable, Union
import numpy as np
from scipy.sparse import csc_array
from .measure import Measure
from .funcbasis import FunctionBasis
from .refdom import ref_doms
    
class Functional:

    form: Callable

    def __init__(self, form: Callable) -> None:
        self.form = form

    def assemble(self, mea: Measure, **extra_args) -> Union[float, np.ndarray]:
        """
        Assemble a functional. 
        """
        dx_ref = ref_doms[mea.dim].dx
        num_quad = mea.quad_w.size
        data = dx_ref * self.form(mea.x, **extra_args) # (rdim, Ne, num_quad)
        rdim = data.shape[0]
        data = data.reshape(-1, num_quad) @ mea.quad_w
        data = data.reshape(rdim, -1).sum(axis=1) # sum over all elements
        return data if data.size > 1 else data.item()
    
class LinearForm(Functional):

    def __init__(self, form: Callable) -> None:
        super().__init__(form)

    def assemble(self, test_basis: FunctionBasis, mea: Measure = None, **extra_args) -> np.ndarray:
        mea = test_basis.mea if mea is None else mea
        dx_ref = ref_doms[mea.dim].dx
        elem_dof = test_basis.fs.elem_dof
        Ne = elem_dof.shape[1] if isinstance(mea.elem_ix, slice) else mea.elem_ix.size
        rows = np.empty((elem_dof.shape[0], Ne), dtype=np.int32)
        vals = np.empty((elem_dof.shape[0], Ne)) # (num_local_dof, Ne)
        for i in range(elem_dof.shape[0]):
            form_data = self.form(test_basis.data[i], mea.x, **extra_args) # (1, Ne, Nq)
            assert form_data.shape[0] == 1, "Please make sure that the form outputs a array of leading dimension 1. "
            vals[i] = dx_ref * (form_data[0] @ mea.quad_w).reshape(-1) # (Ne,), reduce by quadrature
            rows[i] = elem_dof[i, test_basis.mea.elem_ix]
        vec = np.bincount(rows.reshape(-1), weights=vals.reshape(-1), minlength=test_basis.fs.num_dof)
        return vec
    
class BilinearForm(Functional):

    def __init__(self, form: Callable) -> None:
        super().__init__(form)

    def assemble(self, test_basis: FunctionBasis, trial_basis: FunctionBasis, test_mea: Measure = None, trial_mea: Measure = None, **extra_args) -> csc_array:
        test_mea = test_basis.mea if test_mea is None else test_mea
        trial_mea = trial_basis.mea if trial_mea is None else trial_mea
        assert test_mea.dim == trial_mea.dim == test_basis.mea.dim == trial_basis.mea.dim, "The measures must have the same dimension. "
        assert test_basis.data[0].shape[2] == trial_basis.data[0].shape[2], "The number of quadrature points must be the same. "
        dx_ref = ref_doms[test_mea.dim].dx
        test_elem_dof = test_basis.fs.elem_dof
        trial_elem_dof = trial_basis.fs.elem_dof
        test_Ne = test_elem_dof.shape[1] if isinstance(test_basis.mea.elem_ix, slice) else test_basis.mea.elem_ix.size
        trial_Ne = trial_elem_dof.shape[1] if isinstance(trial_basis.mea.elem_ix, slice) else trial_basis.mea.elem_ix.size
        test_2s, trial_2s = test_mea.doubleSided, trial_mea.doubleSided
        assert test_Ne * (1 + trial_2s) == trial_Ne * (1 + test_2s), \
            "Number of elements (facets) in test and trial basis must be the same."
        test_x = (test_mea.x, ) if not test_2s else test_mea.x.sides()
        trial_x = (trial_mea.x, ) if not trial_2s else trial_mea.x.sides()
        rows = np.empty((test_elem_dof.shape[0], trial_elem_dof.shape[0], 1 + test_2s, 1 + trial_2s, test_x[0].shape[1]), dtype=np.int32)
        cols = np.empty_like(rows, dtype=np.int32)
        vals = np.empty_like(rows, dtype=np.float_)
        for i in range(test_elem_dof.shape[0]):
            test_data = (test_basis.data[i], ) if not test_2s else test_basis.data[i].sides()
            for j in range(trial_elem_dof.shape[0]):
                trial_data = (trial_basis.data[j], ) if not trial_2s else trial_basis.data[j].sides()
                for k in range(1 + test_2s):
                    for l in range(1 + trial_2s):
                        form_data = self.form(test_data[k], trial_data[l], test_x[k], trial_x[l], **extra_args) # (1, num_elem, num_quad)
                        assert form_data.shape[0] == 1, "Please make sure that the form outputs a array of leading dimension 1. "
                        vals[i,j,k,l] = dx_ref * (form_data[0] @ test_mea.quad_w).reshape(-1) # (num_elem,), reduce by quadrature
                        rows[i,j,k,l] = test_elem_dof[i, test_basis.mea.elem_ix.reshape(1 + test_2s, -1)[k]]
                        cols[i,j,k,l] = trial_elem_dof[j, trial_basis.mea.elem_ix.reshape(1 + trial_2s, -1)[l]]
        # end for, for
        mat = csc_array((vals.reshape(-1), (rows.reshape(-1), cols.reshape(-1))), \
                        shape=(test_basis.fs.num_dof, trial_basis.fs.num_dof))
        return mat
    