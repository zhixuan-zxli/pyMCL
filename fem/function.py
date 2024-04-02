from typing import Union, Optional
import numpy as np
from .mesh import Mesh
from .fe import FiniteElement

class QuadData(np.ndarray):
    """
    Discrete function values on quadrature points, (rdim * Ne * Nquad).
    """

    grad: np.ndarray
    inv_grad: np.ndarray
    dx: np.ndarray
    cn: np.ndarray # cell normal
    fn: np.ndarray # facet normal

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
        return obj
    
    def __array_finalize__(self, obj) -> None:
        if obj is None: return
        self.grad = getattr(obj, "grad", None)
        self.inv_grad = getattr(obj, "inv_grad", None)
        self.dx = getattr(obj, "dx", None)
        self.cn = getattr(obj, "cn", None)
        self.fn = getattr(obj, "fn", None)

    def __array_wrap__(self, out_arr, context=None):
        # invalidate the attributes
        return np.array(out_arr)



class CellMeasure:
    """
    Represent the volume measure whose dimension equals the topological dimension of the mesh. 
    """

    mesh: Mesh
    elem_ix: Union[np.ndarray, slice]

    def __init__(self, mesh: Mesh, tags: Optional[tuple[int]] = None) -> None:
        self.mesh = mesh
        if tags is None:
            self.elem_ix = slice(None) # select all the elements
        else:
            # select the elements with the provided tags
            elem_tag = mesh.cell_tag[mesh.tdim]
            flag = np.zeros((elem_tag.shape[0],), dtype=np.bool8)
            for t in tags:
                flag[elem_tag == t] = True
            self.elem_ix = np.nonzero(flag)[0]

class FaceMeasure:
    """
    Represent the surface measure whose dimension is one less than the topological dimension of the mesh. 
    """
    
    mesh: Mesh
    facet_ix: tuple[np.ndarray]
    elem_ix: tuple[np.ndarray]
    facet_id: tuple[np.ndarray]

    def __init__(self, mesh: Mesh, tags: Optional[tuple[int]] = None, interior: bool = False) -> None:
        self.mesh = mesh
        if tags is None:
            self.facet_ix = slice(None)
        else:
            # select the facets with the provided tags
            facet_tag = mesh.cell_tag[mesh.tdim-1]
            flag = np.zeros((facet_tag.shape[0],), dtype=np.bool8)
            for t in tags:
                flag[facet_tag == t] = True
            self.facet_ix = np.nonzero(flag)[0]
        #
        elem_ix = []
        facet_id = []
        for k in range(1+interior):
            elem_ix.append(mesh.inv_bdry[k,0,self.facet_ix])
            facet_id.append(mesh.inv_bdry[k,1,self.facet_ix])
        self.elem_ix = tuple(elem_ix)
        self.facet_id = tuple(facet_id)



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
    
    def _interpolate_cell(self, mea: CellMeasure, quad_tab: np.ndarray, x: Optional[QuadData] = None) -> QuadData:
        tdim, rdim = self.fe.elem.tdim, self.fe.elem.rdim
        elem_dof = self.fe.elem_dof[:, mea.elem_ix]
        Ne = elem_dof.shape[1]
        Nq = quad_tab.shape[1]
        data = np.zeros((rdim, Ne, Nq))
        grad = np.zeros((rdim, tdim, Ne, Nq))
        for i in range(elem_dof.shape[0]):
            temp = self.view(np.ndarray)[elem_dof[i]] # (Ne, )
            # interpolate function values
            basis_data = self.fe.elem._eval_basis(i, quad_tab) # (rdim, Nq)
            data += temp[np.newaxis,:,np.newaxis] * basis_data[:, np.newaxis] # (rdim, Ne, Nq)
            # interpolate the gradients
            grad_data = self.fe.elem._eval_grad(i, quad_tab) # (rdim, elem.tdim, Nq)
            grad_temp = temp[np.newaxis,np.newaxis,:,np.newaxis] \
                * grad_data[:,:,np.newaxis] # (rdim, elem.tdim, Ne, Nq)
            if x is not None:
                grad_temp = np.einsum("ij...,jk...->ik...", grad_temp, x.inv_grad)
            grad += grad_temp
            
        data = QuadData(data)
        data.grad = grad
        return data
    
    def _interpolate_facet(self, mea: FaceMeasure, quad_tab: np.ndarray, x: Optional[QuadData] = None) -> QuadData:
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
            for i in range(elem_dof.shape[0]):
                temp = self.view(np.ndarray)[elem_dof[i]] # (Ne, )
                # interpolate function values
                basis_data = self.fe.elem._eval_basis(i, quad_tab.reshape(tdim, -1)).reshape(rdim, -1, Nq) # (rdim, num_facet, Nq)
                data += temp[np.newaxis] * basis_data[:, facet_id, :]
                # interpolate the gradients
                grad_data = self.fe.elem._eval_grad(i, quad_tab.reshape(tdim, -1)).reshape(rdim, tdim, -1, Nq) 
                grad_temp = temp[np.newaxis, np.newaxis] * grad_data[:,:,facet_id,:] # (rdim, tdim, Ne, Nq)
                if x is not None:
                    grad_temp = np.einsum("ij...,jk...->ik...", grad_temp, x.inv_grad)
                # interpolate other things
                if x is None:
                    pass
            res.append(data)
        return res if len(res) > 1 else res[0]
    
class MeshMapping(Function):

    def __new__(cls, fe: FiniteElement):
        obj = np.zeros((fe.num_dof,)).view(cls)
        obj.fe = fe
        return obj
    
    def _interpolate_cell(self, mea: CellMeasure, quad_tab: np.ndarray) -> QuadData:
        data = super()._interpolate_cell(mea, quad_tab, None) 
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
            elif rdim == 2:
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
            elif rdim == 3:
                data.inv_grad[0] = np.cross(data.grad[:,1], data.cn, axis=0) / data.dx
                data.inv_grad[1] = np.cross(data.cn, data.grad[:,0], axis=0) / data.dx
        



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
