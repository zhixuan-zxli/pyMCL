from typing import Optional
import numpy as np
from .mesh import Mesh
from .funcspace import FunctionSpace
from .measure import Measure

class QuadData(np.ndarray):
    """
    Discrete function values on quadrature points, (rdim * Ne * Nquad).
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
        quad_tab = mea.quad_tab
        Nq = quad_tab.shape[1]
        x = mea.x
        # interpolate on cells
        if mea.dim == mea.mesh.tdim:
            elem_dof = self.fe.elem_dof[:, mea.elem_ix]
            Ne = elem_dof.shape[1]
            data = np.zeros((rdim, Ne, Nq))
            grad = np.zeros((rdim, tdim, Ne, Nq))
            for i in range(elem_dof.shape[0]): # loop over each basis function
                temp = self.view(np.ndarray)[elem_dof[i]] # (Ne, )
                basis_data, grad_data = self.fe.elem._eval(i, quad_tab) # (rdim, Nq), (rdim, tdim, Nq)
                # interpolate function values
                data += temp[np.newaxis,:,np.newaxis] * basis_data[:, np.newaxis] # (rdim, Ne, Nq)
                # interpolate the gradients
                grad_temp = temp[np.newaxis,np.newaxis,:,np.newaxis] \
                    * grad_data[:,:,np.newaxis] # (rdim, elem.tdim, Ne, Nq)
                if x is not None:
                    grad_temp = np.einsum("ij...,jk...->ik...", grad_temp, x.inv_grad)
                grad += grad_temp
            #
            data = QuadData(data)
            data.grad = grad
            return data
        if mea.dim == mea.mesh.tdim-1:
            # transform the quadrature locations via facet_id here
            quad_tab = self.fe.elem.ref_cell._broadcast_facet(quad_tab) # (tdim, num_facet, Nq)
            res = []
            for elem_ix, facet_id, y in zip(mea.elem_ix, mea.facet_id, x):
                # facet_id: (Ne, )
                elem_dof = self.fe.elem_dof[:, elem_ix]
                Ne = elem_dof.shape[1]
                data = np.zeros((rdim, Ne, Nq))
                grad = np.zeros((rdim, tdim, Ne, Nq))
                for i in range(elem_dof.shape[0]):
                    temp = self.view(np.ndarray)[elem_dof[i]] # (Ne, )
                    # interpolate function values
                    basis_data, grad_data = self.fe.elem._eval(i, quad_tab.reshape(tdim, -1)) 
                    basis_data = basis_data.reshape(rdim, -1, Nq) # (rdim, num_facet, Nq)
                    data += temp[np.newaxis] * basis_data[:, facet_id, :] # (rdim, Ne, Nq)
                    # interpolate the gradients
                    grad_data = grad_data.reshape(rdim, tdim, -1, Nq)
                    grad_temp = temp[np.newaxis, np.newaxis] * grad_data[:,:,facet_id,:] # (rdim, tdim, Ne, Nq)
                    if y is not None:
                        grad_temp = np.einsum("ij...,jk...->ik...", grad_temp, y.inv_grad)
                    grad += grad_temp
                data = QuadData(data)
                data.grad = grad
                res.append(data)
            return tuple(res)
        #
        raise RuntimeError("Incorrect measure dimension")

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
