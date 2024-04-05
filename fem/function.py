from typing import Union, Optional
import numpy as np
from .mesh import Mesh
from .funcspace import FunctionSpace
from .measure import CellMeasure, FaceMeasure

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
    
    def _interpolate_cell(self, mea: CellMeasure, quad_tab: np.ndarray, x: QuadData) -> QuadData:
        assert self.fe.mesh is mea.mesh
        tdim, rdim = self.fe.elem.tdim, self.fe.elem.rdim
        elem_dof = self.fe.elem_dof[:, mea.elem_ix]
        Ne = elem_dof.shape[1]
        Nq = quad_tab.shape[1]
        data = np.zeros((rdim, Ne, Nq))
        grad = np.zeros((rdim, tdim, Ne, Nq))
        for i in range(elem_dof.shape[0]):
            temp = self.view(np.ndarray)[elem_dof[i]] # (Ne, )
            basis_data, grad_data = self.fe.elem._eva(i, quad_tab) # (rdim, Nq)
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
    
    def _interpolate_facet(self, mea: FaceMeasure, quad_tab: np.ndarray, x: QuadData) -> tuple[QuadData]:
        assert self.fe.mesh is mea.mesh
        tdim, rdim = self.fe.elem.tdim, self.fe.elem.rdim
        Nq = quad_tab.shape[1]
        # transform the quadrature locations via facet_id here
        quad_tab = self.fe.elem.ref_cell._broadcast_facet(quad_tab) # (tdim, num_facet, Nq)
        res = []
        for elem_ix, facet_id in zip(mea.elem_ix, mea.facet_id):
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
                if x is not None:
                    grad_temp = np.einsum("ij...,jk...->ik...", grad_temp, x.inv_grad)
                grad += grad_temp
            data = QuadData(data)
            data.grad = grad
            res.append(data)
        return tuple(res)
    
class MeshMapping(Function):

    def __new__(cls, fe: FunctionSpace):
        obj = np.zeros((fe.num_dof,)).view(cls)
        obj.fe = fe
        return obj
    
    def _interpolate_cell(self, mea: CellMeasure, quad_tab: np.ndarray) -> QuadData:
        data = super()._interpolate_cell(mea, quad_tab, x=None) 
        tdim, rdim = self.fe.elem.tdim, self.fe.elem.rdim
        assert tdim == rdim or tdim == rdim-1
        Ne, Nq = data.shape[1:]
        # grad: (rdim, tdim, Ne, Nq)
        # build dx: (1, Ne, Nq)
        # build cn: (rdim, Ne, Nq)
        # build inv_grad: (tdim, rdim, Ne, Nq)
        if tdim == 0:
            data.dx = np.ones((1, Ne, Nq))
            data.cn = np.ones((rdim, Ne, Nq)) if rdim > tdim else None
            data.inv_grad = np.zeros((0, rdim, Ne, Nq))
        elif tdim == 1:
            data.dx = np.linalg.norm(data.grad, ord=None, axis=0)
            if rdim == 1:
                data.inv_grad = 1.0 / data.grad
            else:
                t = data.grad[:,0] / data.dx # (2, Ne, Nq)
                data.cn = np.array((t[1], -t[0]))
                data.inv_grad = data.grad[:,0] / data.dx**2
                data.inv_grad = data.inv_grad[np.newaxis]
        elif tdim == 2:
            data.dx = np.cross(data.grad[:,0], data.grad[:,1], axis=0) # (Ne,Nq) or (3,Ne,Nq)
            if rdim == 3:
                data.cn = data.dx
                data.dx = np.linalg.norm(data.dx, ord=None, axis=0) #(Ne, Nq)
                data.cn = data.cn / data[np.newaxis]
            data.dx = data.dx[np.newaxis]
            data.inv_grad = np.zeros((tdim, rdim, Ne, Nq))
            if rdim == 2:
                data.inv_grad[0,0] = data.grad[1,1] / data.dx[0]
                data.inv_grad[0,1] = -data.grad[0,1] / data.dx[0]
                data.inv_grad[1,0] = -data.grad[1,0] / data.dx[0]
                data.inv_grad[1,1] = data.grad[0,0] / data.dx[0]
            else:
                data.inv_grad[0] = np.cross(data.grad[:,1], data.cn, axis=0) / data.dx
                data.inv_grad[1] = np.cross(data.cn, data.grad[:,0], axis=0) / data.dx
        return data

    def _interpolate_facet(self, mea: FaceMeasure, quad_tab: np.ndarray) -> tuple[QuadData]:
        data_tuple = super()._interpolate_facet(mea, quad_tab, x=None)
        tdim, rdim = self.fe.elem.tdim, self.fe.elem.rdim
        assert tdim == rdim or tdim == rdim-1
        Ne, Nq = data.shape[1:]
        ref_fn = self.fe.elem.ref_cell.facet_normal # (tdim, num_facet)
        for data, facet_id in zip(data_tuple, mea.facet_id):
            # build dx
            # build cn
            # build inv_grad: (tdim, rdim, Ne, Nq)
            # build fn: (rdim, Ne, Nq)
            # build ds
            if tdim == 0:
                data.dx = np.ones((1, Ne, Nq))
                data.cn = np.ones((rdim, Ne, Nq)) if rdim > tdim else None
                data.inv_grad = np.zeros((0, rdim, Ne, Nq))
            elif tdim == 1:
                data.dx = np.linalg.norm(data.grad, ord=None, axis=0)
                if rdim == 1:
                    data.inv_grad = 1.0 / data.grad
                else:
                    t = data.grad[:,0] / data.dx # (2, Ne, Nq)
                    data.cn = np.array((t[1], -t[0]))
                    data.inv_grad = data.grad[:,0] / data.dx**2
                    data.inv_grad = data.inv_grad[np.newaxis]
            elif tdim == 2:
                data.dx = np.cross(data.grad[:,0], data.grad[:,1], axis=0) # (Ne,Nq) or (3,Ne,Nq)
                if rdim == 3:
                    data.cn = data.dx
                    data.dx = np.linalg.norm(data.dx, ord=None, axis=0) #(Ne, Nq)
                    data.cn = data.cn / data[np.newaxis]
                data.dx = data.dx[np.newaxis]
                data.inv_grad = np.zeros((tdim, rdim, Ne, Nq))
                if rdim == 2:
                    data.inv_grad[0,0] = data.grad[1,1] / data.dx[0]
                    data.inv_grad[0,1] = -data.grad[0,1] / data.dx[0]
                    data.inv_grad[1,0] = -data.grad[1,0] / data.dx[0]
                    data.inv_grad[1,1] = data.grad[0,0] / data.dx[0]
                else:
                    data.inv_grad[0] = np.cross(data.grad[:,1], data.cn, axis=0) / data.dx
                    data.inv_grad[1] = np.cross(data.cn, data.grad[:,0], axis=0) / data.dx
            # build the facet normal and surface Jacobian
            data.fn = np.sum(data.inv_grad * ref_fn[:, np.newaxis, facet_id, np.newaxis], axis=0) # (rdim, Ne, Nq)
            nm = np.linalg.norm(data.fn, ord=None, axis=0, keepdims=True) # (1, Ne, Nq)
            data.fn = data.fn / nm
            data.ds = data.dx * nm
        return data_tuple
        

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
