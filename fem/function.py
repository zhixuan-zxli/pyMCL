from typing import Union, Optional
import numpy as np
from .mesh import Mesh
from .fe import FiniteElement

class FieldData(np.ndarray):
    """
    Discrete function values on quadrature points, (rdim * Ne * Nquad).
    """
    def __new__(cls, value: Optional[np.ndarray] = None):
        if value is None:
            obj = np.array((0,)).view(cls)
        else:
            obj = value.view(cls)
        obj.grad = None
        obj.dx = None
        obj.n = None
        obj.inv_grad = None
        return obj
    
    def __array_finalize__(self, obj) -> None:
        if obj is None: return
        self.grad = getattr(obj, "grad", None)
        self.dx = getattr(obj, "dx", None)
        self.n = getattr(obj, "n", None)
        self.inv_grad = getattr(obj, "inv_grad", None)

    def __array_wrap__(self, out_arr, context=None):
        # invalidate the attributes
        return np.array(out_arr)

class Measure:

    mesh: Mesh
    tdim: int
    elem_ix: Union[np.ndarray, slice]
    # facet_ix: Optional[np.ndarray]

    def __init__(self, mesh: Mesh, tdim: int, ix = None) -> None:
        self.mesh = mesh
        self.tdim = tdim
        if ix is not None:
            self.elem_ix = ix
        else:
            self.elem_ix = slice(None)


class Function(np.ndarray):
    """
    Array of size num_dof. 
    """
    fe: FiniteElement
    
    def __new__(cls, fe: FiniteElement):
        obj = np.zeros((fe.num_dof,)).view(cls)
        obj.fe = fe
        return obj
    
    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self.fe = getattr(obj, "fe", None)

    def __array_wrap__(self, out_arr, context=None):
        out_arr.fe = self.fe
        return out_arr
    
    def update(self) -> None:
        # Update self to account for periodic BC. 
        if self.fe.periodic:
            raise NotImplementedError
    
    def interpolate(self, mea: Measure, quad_tab: np.ndarray, x: Optional[FieldData] = None) -> FieldData:
        tdim = mea.tdim
        rdim = self.fe.elem.rdim
        elem_dof = self.fe.elem_dof[:, mea.elem_ix]
        Ne = elem_dof.shape[1]
        Nq = quad_tab.shape[1]
        data = np.zeros((rdim, Ne, Nq))
        grad = np.zeros((rdim, tdim, Ne, Nq))
        for i in range(elem_dof.shape[0]):
            temp = self.view(np.ndarray)[elem_dof[i]] # (Ne, )
            # interpolate function values
            basis_data = self.fe.elem._eval_basis(i, quad_tab) # (rdim, Nq)
            data += temp[:, np.newaxis] * basis_data[:, np.newaxis] # (rdim, Ne, Nq)
            # interpolate the gradients
            grad_data = self.fe.elem._eval_grad(i, quad_tab) # (rdim, elem.tdim, Nq)
            grad_temp = temp[:, :, np.newaxis] * grad_data[:,:,np.newaxis] # (rdim, elem.tdim, Ne, Nq)
            if x is not None:
                grad_temp = np.einsum("ij...,jk...->ik...", grad_temp, x.inv_grad)
            grad += grad_temp
            # interpolate other things ...
            if x is None:
                pass
        data = FieldData(data)
        data.grad = grad
        return data


# =============================================================
    
def group_fn(*fnlist: Function) -> np.ndarray:
    v_size = sum(f.size for f in fnlist)
    vec = np.zeros((v_size, ))
    index = 0
    for f in fnlist:
        vec[index:index+f.size] = f
        index += f.size
    return vec

def split_fn(vec: np.ndarray, *fnlist: Function) -> None:
    index = 0
    for f in fnlist:
        f[:] = vec[index:index+f.size]
        index += f.size
    assert index == vec.size
