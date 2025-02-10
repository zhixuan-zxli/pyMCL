from typing import Callable, Union
import numpy as np
from scipy.sparse import csr_array
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
        Nq = mea.quad_w.size
        data = dx_ref * self.form(mea.x, **extra_args) # (rdim, Ne, num_quad)
        rdim = data.shape[0]
        data = data.reshape(-1, Nq) @ mea.quad_w
        data = data.reshape(rdim, -1).sum(axis=1) # sum over all elements
        return data if data.size > 1 else data.item()
    
class LinearForm(Functional):

    def __init__(self, form: Callable) -> None:
        super().__init__(form)

    def assemble(self, test_basis: FunctionBasis, mea: Measure, **extra_args) -> np.ndarray:
        dx_ref = ref_doms[mea.dim].dx
        elem_dof = test_basis.fs.elem_dof
        Ne = elem_dof.shape[1] if isinstance(mea.elem_ix, slice) else mea.elem_ix.size
        if not mea.interiorFacet:
            rows = np.empty((elem_dof.shape[0], Ne), dtype=np.int32)
            vals = np.empty((elem_dof.shape[0], Ne)) # (num_local_dof, Ne)
            for i in range(elem_dof.shape[0]):
                form_data = self.form(test_basis.data[i], mea.x, **extra_args) # (1, Ne, Nq)
                assert form_data.shape[0] == 1
                vals[i] = dx_ref * (form_data[0] @ mea.quad_w).reshape(-1) # (Ne,), reduce by quadrature
                rows[i] = elem_dof[i, test_basis.mea.elem_ix]
        else: # interior facets need special treatment
            assert Ne % 2 == 0
            Nf = Ne // 2
            rows = np.empty((elem_dof.shape[0], 2, Nf), dtype=np.int32)
            vals = np.empty((elem_dof.shape[0], 2, Nf))
            x_sides = mea.x.sides()
            for i in range(elem_dof.shape[0]):
                test_data = test_basis.data[i].sides()
                for m in (0, 1):
                    form_data = self.form(test_data[m], x_sides[m], **extra_args) # (1, Nf, Nq)
                    vals[i,m] = dx_ref * (form_data[0] @ mea.quad_w).reshape(-1) # (Ne,), reduce by quadrature
                    rows[i,m] = elem_dof[i, test_basis.mea.elem_ix.reshape(2,-1)[m]]
        vec = np.bincount(rows.reshape(-1), weights=vals.reshape(-1), minlength=test_basis.fs.num_dof)
        return vec
    
class BilinearForm(Functional):

    def __init__(self, form) -> None:
        super().__init__(form)

    def assemble(self, test_basis: FunctionBasis, trial_basis: FunctionBasis, mea: Measure, **extra_args) -> csr_array:
        assert mea.dim == test_basis.mea.dim and mea.dim == trial_basis.mea.dim
        dx_ref = ref_doms[mea.dim].dx
        test_elem_dof = test_basis.fs.elem_dof
        trial_elem_dof = trial_basis.fs.elem_dof
        test_Ne = test_elem_dof.shape[1] if isinstance(test_basis.mea.elem_ix, slice) else test_basis.mea.elem_ix.size
        trial_Ne = trial_elem_dof.shape[1] if isinstance(trial_basis.mea.elem_ix, slice) else trial_basis.mea.elem_ix.size
        assert test_Ne == trial_Ne
        if not mea.interiorFacet:
            rows = np.empty((test_elem_dof.shape[0], trial_elem_dof.shape[0], test_Ne), dtype=np.int32)
            cols = np.empty((test_elem_dof.shape[0], trial_elem_dof.shape[0], test_Ne), dtype=np.int32)
            vals = np.empty((test_elem_dof.shape[0], trial_elem_dof.shape[0], test_Ne))
            for i in range(test_elem_dof.shape[0]):
                for j in range(trial_elem_dof.shape[0]):
                    form_data = self.form(test_basis.data[i], trial_basis.data[j], mea.x, **extra_args) # (1, Ne, Nq)
                    vals[i,j] = dx_ref * (form_data[0] @ mea.quad_w).reshape(-1) # (Ne,), reduce by quadrature
                    rows[i,j] = test_elem_dof[i, test_basis.mea.elem_ix]
                    cols[i,j] = trial_elem_dof[j, trial_basis.mea.elem_ix]
            # end for, for
        else: # interior facets need special treatment
            assert test_Ne % 2 == 0
            Nf = test_Ne // 2
            rows = np.empty((test_elem_dof.shape[0], trial_elem_dof.shape[0], 2, 2, Nf), dtype=np.int32)
            cols = np.empty((test_elem_dof.shape[0], trial_elem_dof.shape[0], 2, 2, Nf), dtype=np.int32)
            vals = np.empty((test_elem_dof.shape[0], trial_elem_dof.shape[0], 2, 2, Nf))
            x_sides = mea.x.sides()
            for i in range(test_elem_dof.shape[0]):
                v_sides = test_basis.data[i].sides()
                for j in range(trial_elem_dof.shape[0]):
                    u_sides = trial_basis.data[j].sides()
                    for m, n in (0,0), (0,1), (1,0), (1,1):
                        form_data = self.form(v_sides[m], u_sides[n], x_sides[m], x_sides[n], **extra_args) # (1, Nf, Nq)
                        vals[i,j,m,n] = dx_ref * (form_data[0] @ mea.quad_w).reshape(-1) # (Nf,), reduce by quadrature
                        rows[i,j,m,n] = test_elem_dof[i, test_basis.mea.elem_ix.reshape(2,-1)[m]]
                        cols[i,j,m,n] = trial_elem_dof[j, trial_basis.mea.elem_ix.reshape(2,-1)[n]]
            # end for, for
        #
        mat = csr_array((vals.reshape(-1), (rows.reshape(-1), cols.reshape(-1))), \
                        shape=(test_basis.fs.num_dof, trial_basis.fs.num_dof))
        return mat
    