from typing import Optional
import numpy as np
from .funcspace import FunctionSpace
from .measure import Measure

class QuadData(np.ndarray):
    """
    Discrete function values on quadrature points, (rdim, num_elem, num_quad).
    """

    grad: np.ndarray     # of shape (rdim, tdim, num_elem, num_quad)
    inv_grad: np.ndarray # of shape (tdim, rdim, num_elem, num_quad)
    dx: np.ndarray       # of shape (1, num_elem, num_quad)
    cn: np.ndarray       # cell normal, (gdim, num_elem, num_quad)
    fn: np.ndarray       # facet normal, (gdim, num_elem, num_quad), and points to the element with the bigger tag and is irrelevant to the facet orientation
    ds: np.ndarray       # surface Jacobian (1, num_elem, num_quad)

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

    # Split the num_elem dimension into two parts. Useful when dealing with discontinuous elements. 
    def sides(self) -> tuple["QuadData", "QuadData"]:
        assert self.shape[1] % 2 == 0
        Nf = self.shape[1] // 2
        u1, u2 = QuadData(self[:,:Nf,:]), QuadData(self[:,Nf:,:])
        u1.grad, u2.grad = np.split(self.grad, 2, axis=2)
        if self.dx is not None:
            u1.dx, u2.dx = np.split(self.dx, 2, axis=1)
        if self.cn is not None:
            u1.cn, u2.cn = np.split(self.cn, 2, axis=1)
        if self.fn is not None:
            u1.fn, u2.fn = np.split(self.fn, 2, axis=1)
        if self.ds is not None:
            u1.ds, u2.ds = np.split(self.ds, 2, axis=1)
        # omit inv_grad
        return u1, u2


# ===============================================================================


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
    
    def _interpolate(self, mea: Measure) -> QuadData:
        assert self.fe.mesh is mea.mesh
        tdim, rdim = self.fe.elem.tdim, self.fe.elem.rdim
        num_quad = mea.quad_w.size
        elem_dof = self.fe.elem_dof
        Ne = elem_dof.shape[1] if isinstance(mea.elem_ix, slice) else mea.elem_ix.size # num_elem if volume measure or n * num_facets if surface measure
        gdim = tdim if mea.x is None else mea.mesh.gdim
        data = np.zeros((rdim, Ne, num_quad))
        grad = np.zeros((rdim, gdim, Ne, num_quad))
        # interpolate on cells
        if mea.dim == tdim:
            for i in range(elem_dof.shape[0]): # loop over each basis function
                nodal = self.view(np.ndarray)[elem_dof[i, mea.elem_ix]] # (Ne, )
                basis_data, grad_data = self.fe.elem._eval(i, mea.quad_tab.T) # (rdim, num_quad), (rdim, tdim, num_quad)
                # interpolate function values
                data += nodal[np.newaxis,:,np.newaxis] * basis_data[:, np.newaxis, :] # (rdim, Ne, num_quad)
                # interpolate the gradients
                grad_temp = nodal[np.newaxis,np.newaxis,:,np.newaxis] \
                    * grad_data[:,:,np.newaxis,:] # (rdim, gdim, Ne, num_quad)
                if mea.x is not None:
                    grad_temp = np.einsum("ij...,jk...->ik...", grad_temp, mea.x.inv_grad)
                grad += grad_temp
        elif mea.dim == tdim-1:
            for i in range(elem_dof.shape[0]):
                nodal = self.view(np.ndarray)[elem_dof[i, mea.elem_ix]] # (n*num_facets, )
                # interpolate function values
                basis_data, grad_data = self.fe.elem._eval(i, mea.quad_tab.reshape(-1, tdim).T)  # quad_tab is of shape (n * num_facets, num_quad, tdim)
                basis_data = basis_data.reshape(rdim, -1, num_quad) # (rdim, n*num_facets, num_quad)
                data += nodal[np.newaxis,:,np.newaxis] * basis_data # (rdim, n*num_facets, num_quad)
                # interpolate the gradients
                grad_data = grad_data.reshape(rdim, tdim, -1, num_quad) # (rdim, tdim, n*num_facets, num_quad)
                grad_temp = nodal[np.newaxis,np.newaxis,:,np.newaxis] * grad_data # (rdim, tdim, n*num_facets, num_quad)
                if mea.x is not None:
                    grad_temp = np.einsum("ij...,jk...->ik...", grad_temp, mea.x.inv_grad) # (rdim, gdim, n*num_facets, num_quad)
                grad += grad_temp
        else:
            raise RuntimeError("Incorrect measure dimension. ")
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
