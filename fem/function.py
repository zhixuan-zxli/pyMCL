from typing import Optional
import numpy as np
from fe import FiniteElement

class QuadData(np.ndarray):
    """
    Array of size rdim * Ne * Nquad, for assembly. 
    Attributes: 
    grad rdim * tdim * Ne * Nquad
    inv_grad
    dx
    n
    """

    def __new__(cls, value : Optional[np.ndarray]):
        if value is None:
            obj = np.array([0]).view(cls)
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



class Function(np.ndarray):
    """
    Array of size num_dof. 
    """
    fe: FiniteElement
    
    def __new__(cls, fe: FiniteElement):
        obj = np.zeros((fe.num_dof, fe.num_copy)).view(cls)
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
            temp = self.copy()
            self[:] = temp[self.fe.dof_remap]
    
    def _get_quad_data(self, basis_type, cell_dof: np.ndarray, x: Optional[QuadData], quadTable: np.ndarray, hint) -> QuadData:
        """
        See the doc of QuadData for hints. 
        """
        tdim = basis_type.tdim
        num_dof_per_elem = basis_type.num_dof_per_elem
        rdim = max(self.fe.num_copy, self.fe.rdim)
        data = None
        if "f" in hint:
            data = np.zeros((rdim, cell_dof.shape[0], quadTable.shape[1]))
            for i in range(num_dof_per_elem):
                basis = basis_type._eval_basis(i, quadTable) # (rdim, Nq)
                temp = np.array(self)[cell_dof[:,i], :].T
                data = data + temp[:,:,np.newaxis] * basis[:, np.newaxis, :] # (rdim, Ne, Nq)
        data = QuadData(data)
        if "grad" in hint:
            grad = np.zeros((rdim, tdim, cell_dof.shape[0], quadTable.shape[1])) # (rdim, tdim, Ne, Nq)
            for i in range(num_dof_per_elem):
                basis_grad = basis_type._eval_grad(i, quadTable) # (rdim, tdim, Nq)
                temp = np.array(self)[cell_dof[:,i], :].T
                grad = grad + temp[:,np.newaxis,:,np.newaxis] * basis_grad[:, :, np.newaxis, :]
            if x is not None:
                grad = np.einsum("ij...,jk...->ik...", grad, x.inv_grad)
            data.grad = grad # (rdim, gdim, Ne, Nq)
        if "dx" in hint:
            assert x is None
            if tdim == 1:
                data.dx = np.linalg.norm(grad[:,0,:,:], axis=0)
                data.dx = data.dx[np.newaxis] # (1, Ne, Nq)
            elif tdim == 2:
                data.dx = np.cross(grad[:, 0, :, :], grad[:, 1, :, :], axis=0)
                if rdim == 3:
                    data.dx = np.linalg.norm(data.dx, ord=None, axis=0) # (Ne, Nq)
                data.dx = data.dx[np.newaxis] # (1, Ne, Nq)
            else:
                raise NotImplementedError
        if "n" in hint:
            assert x is None
            if tdim == 1:
                assert(rdim == 2)
                t = np.squeeze(grad) / data.dx
                data.n = np.zeros_like(t)
                data.n[0] = t[1]
                data.n[1] = -t[0]
            elif tdim == 2:
                assert(rdim == 3)
                temp = np.cross(grad[:, 0, :, :], grad[:, 1, :, :], axis=0) # (3, Ne, Nq)
                data.n = temp / data.dx
            else:
                raise RuntimeError("Cannot calculate the normal of a 3D cell. ")
        if "inv_grad" in hint:
            assert x is None
            if tdim == 1 and rdim == 1:
                data.inv_grad = 1.0 / data.grad
            elif tdim == 1 and rdim == 2:
                temp = np.squeeze(data.grad) # (2, Ne, Nq)
                data.inv_grad = (temp / data.dx**2).reshape(1, 2, data.grad.shape[2], data.grad.shape[3])
            elif tdim == 2 and rdim == 2:
                data.inv_grad = np.zeros_like(data.grad)
                data.inv_grad[0, 0, :, :] = data.grad[1, 1, :, :] / data.dx[0,:,:]
                data.inv_grad[0, 1, :, :] = -data.grad[0, 1, :, :] / data.dx[0,:,:]
                data.inv_grad[1, 0, :, :] = -data.grad[1, 0, :, :] / data.dx[0,:,:]
                data.inv_grad[1, 1, :, :] = data.grad[0, 0, :, :] / data.dx[0,:,:]
            elif tdim == 2 and rdim == 3:
                data.inv_grad = np.zeros((2, 3, data.grad.shape[2], data.grad.shape[3]))
                data.inv_grad[0, :, :, :] = np.cross(data.grad[:,1,:,:], data.n, axis=0) / data.dx
                data.inv_grad[1, :, :, :] = np.cross(data.n, data.grad[:,0,:,:], axis=0) / data.dx
            else:
                raise RuntimeError("Unable to calculate inv_grad for tdim={} and rdim={}.".format(self.fe.tdim, rdim))
        # todo: conormal
        return data


# =============================================================
    
def group_fn(*fnlist: Function) -> np.ndarray:
    v_size = sum(f.size for f in fnlist)
    vec = np.zeros((v_size, ))
    index = 0
    for f in fnlist:
        vec[index:index+f.size] = f.reshape(-1)
        index += f.size
    return vec

def split_fn(vec: np.ndarray, *fnlist: Function) -> None:
    index = 0
    for f in fnlist:
        f[:] = vec[index:index+f.size].reshape(f.shape)
        index += f.size
    assert index == vec.size
