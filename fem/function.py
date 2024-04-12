from typing import Optional
import numpy as np
from .funcspace import FunctionSpace
from .measure import Measure

class QuadData(np.ndarray):
    """
    Discrete function values on quadrature points, (rdim, Ne, Nquad).
    """

    grad: np.ndarray
    inv_grad: np.ndarray
    dx: np.ndarray
    cn: np.ndarray # cell normal
    fn: np.ndarray # facet normal
    ds: np.ndarray # surface Jacobian

    def __new__(cls, value: Optional[np.ndarray] = None):
        if value is None:
            obj = np.array((0,)).view(cls)
        else:
            obj = value.view(cls)
        obj.grad = None
        obj.inv_grad = None
        obj.dx = None
        obj.cn = None
        obj.fn = None
        obj.ds = None
        return obj
    
    def __array_finalize__(self, obj) -> None:
        if obj is None: return
        self.grad = getattr(obj, "grad", None)
        self.inv_grad = getattr(obj, "inv_grad", None)
        self.dx = getattr(obj, "dx", None)
        self.cn = getattr(obj, "cn", None)
        self.fn = getattr(obj, "fn", None)
        self.ds = getattr(obj, "ds", None)

    def __array_wrap__(self, out_arr, context=None):
        # invalidate the attributes
        return np.array(out_arr)


class Function(np.ndarray):
    """
    Array of size num_dof. 
    """
    fe: FunctionSpace
    
    def __new__(cls, fe: FunctionSpace):
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
    
    def _interpolate(self, mea: Measure) -> QuadData:
        assert self.fe.mesh is mea.mesh
        tdim, rdim = self.fe.elem.tdim, self.fe.elem.rdim
        Nq = mea.quad_w.size
        elem_dof = self.fe.elem_dof
        Ne = elem_dof.shape[1] if isinstance(mea.elem_ix, slice) else mea.elem_ix.size
        gdim = tdim if mea.x is None else mea.mesh.gdim
        data = np.zeros((rdim, Ne, Nq))
        grad = np.zeros((rdim, gdim, Ne, Nq))
        # interpolate on cells
        if mea.dim == tdim:
            for i in range(elem_dof.shape[0]): # loop over each basis function
                nodal = self.view(np.ndarray)[elem_dof[i, mea.elem_ix]] # (Ne, )
                basis_data, grad_data = self.fe.elem._eval(i, mea.quad_tab.T) # (rdim, Nq), (rdim, tdim, Nq)
                # interpolate function values
                data += nodal[np.newaxis,:,np.newaxis] * basis_data[:, np.newaxis, :] # (rdim, Ne, Nq)
                # interpolate the gradients
                grad_temp = nodal[np.newaxis,np.newaxis,:,np.newaxis] \
                    * grad_data[:,:,np.newaxis,:] # (rdim, gdim, Ne, Nq)
                if mea.x is not None:
                    grad_temp = np.einsum("ij...,jk...->ik...", grad_temp, mea.x.inv_grad)
                grad += grad_temp
        elif mea.dim == tdim-1:
            for i in range(elem_dof.shape[0]):
                nodal = self.view(np.ndarray)[elem_dof[i, mea.elem_ix]] # (Nf, )
                # interpolate function values
                basis_data, grad_data = self.fe.elem._eval(i, mea.quad_tab.reshape(-1, tdim).T) 
                basis_data = basis_data.reshape(rdim, -1, Nq) # (rdim, Nf, Nq)
                data += nodal[np.newaxis,:,np.newaxis] * basis_data # (rdim, Nf, Nq)
                # interpolate the gradients
                grad_data = grad_data.reshape(rdim, tdim, -1, Nq) # (rdim, tdim, Nf, Nq)
                grad_temp = nodal[np.newaxis,np.newaxis,:,np.newaxis] * grad_data # (rdim, tdim, Nf, Nq)
                if mea.x is not None:
                    grad_temp = np.einsum("ij...,jk...->ik...", grad_temp, mea.x.inv_grad) # (rdim, gdim, Nf, Nq)
                grad += grad_temp
        else:
            raise RuntimeError("Incorrect measure dimension")
        #
        data = QuadData(data)
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
