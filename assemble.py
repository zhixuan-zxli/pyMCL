from typing import Optional
import numpy as np
from scipy.sparse import csr_matrix
from mesh import *
from fe import *
from quadrature import *


class Measure:
    def __init__(self, mesh: Mesh, tdim: int, sub_id: Optional[int] = None) -> None:
        self.mesh = mesh
        self.tdim = tdim
        self.sub_id = sub_id

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
            obj = np.array([0])
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
    # is this function is a mapping, then there is no chain rule when evaluating gradients. 
    isMeshMapping: bool
    
    def __new__(cls, fe: FiniteElement, isMeshMapping: bool = False):
        obj = np.zeros((fe.num_copy, fe.num_dof)).view(cls)
        obj.fe = fe
        obj.isMeshMapping = isMeshMapping
        return obj
    
    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self.fe = getattr(obj, "fe", None)
        self.isMeshMapping = getattr(obj, "isMeshMapping", False)

    def __array_wrap__(self, out_arr, context=None):
        out_arr.fe = self.fe
        out_arr.isMeshMapping = self.isMeshMapping
        return out_arr
    
    def _get_quad_data(self, hint, mea: Measure, quadTable: np.ndarray) -> QuadData:
        """
        See the doc of QuadData for hints. 
        """
        # check cache
        dof = self.fe.getCellDof(mea.tdim, mea.sub_id)
        basis_provider = type(self.fe)
        if mea.tdim < mea.mesh.tdim:
            basis_provider = basis_provider.trace_type[mea.tdim]
        num_dof_per_elem = basis_provider.num_dof_per_elem
        rdim = np.maximum(self.fe.num_copy, self.fe.rdim)
        data = None
        if "basis" in hint:
            data = np.zeros((rdim, dof.shape[0], quadTable.shape[1]))
            for i in range(num_dof_per_elem):
                basis = basis_provider._eval_basis(i, quadTable) # (rdim, num_quad)
                data = data + self[:, dof[:,i], np.newaxis] * basis[:, np.newaxis, :]
        data = QuadData(data)
        if "grad" in hint:
            grad = np.zeros((rdim, self.fe.tdim, dof.shape[0], quadTable.shape[1])) # (rdim, tdim, Ne, num_quad)
            for i in range(num_dof_per_elem):
                basis_grad = basis_provider._eval_grad(i, quadTable) # (rdim, tdim, num_quad)
                grad = grad + self[:, np.newaxis, dof[:,i], np.newaxis] * basis_grad[:, :, np.newaxis, :]
            if not self.isMeshMapping:
                mapping = self.fe.mesh.mapping._get_quad_data(["grad", "inv_grad"], mea, quadTable) # (tdim, gdim, Ne, num_quad)
                grad = np.einsum("ij...,jk...->ik...", grad, mapping.inv_grad)
            data.grad = grad # (rdim, gdim, Ne, num_quad)
        if "dx" in hint:
            assert(self.isMeshMapping)
            if mea.tdim == 1:
                data.dx = np.sqrt(np.squeeze(np.sum(grad**2, axis=0))) # (Ne, num_quad)
            elif mea.tdim == 2:
                temp = np.cross(grad[:, 0, :, :], grad[:, 1, :, :], axis=0)
                data.dx = np.sqrt(np.squeeze(np.sum(temp**2, axis=0))) # (Ne, num_quad)
            else:
                raise NotImplementedError
        if "n" in hint:
            assert(self.isMeshMapping)
            if mea.tdim == 1:
                assert(rdim == 2)
                t = np.squeeze(grad) / data.dx[np.newaxis, :, :]
                data.n = np.zeros_like(t)
                data.n[0, :] = t[1, :]
                data.n[1, :] = -t[0, :]
            elif mea.tdim == 2:
                assert(rdim == 3)
                temp = np.cross(grad[:, 0, :, :], grad[:, 1, :, :], axis=0) # (3, Ne, num_quad)
                data.n = temp / data.dx[np.newaxis, :, :]
            else:
                raise RuntimeError("Cannot calculate the normal of a 3D cell. ")
        if "inv_grad" in hint:
            assert(self.isMeshMapping)
            if self.fe.tdim == 1 and rdim == 1:
                data.inv_grad = 1.0 / data.grad
            elif self.fe.tdim == 1 and rdim == 2:
                temp = np.squeeze(data.grad) # (2, Ne, num_quad)
                data.inv_grad = (temp / data.dx[np.newaxis, :, :]**2).reshape(1, 2, data.grad.shape[2], data.grad.shape[3])
            elif self.fe.tdim == 2 and rdim == 2:
                data.inv_grad = np.zeros_like(data.grad)
                data.inv_grad[0, 0, :, :] = data.grad[1, 1, :, :] / data.dx
                data.inv_grad[0, 1, :, :] = -data.grad[0, 1, :, :] / data.dx
                data.inv_grad[1, 0, :, :] = -data.grad[1, 0, :, :] / data.dx
                data.inv_grad[1, 1, :, :] = data.grad[0, 0, :, :] / data.dx
            elif self.fe.tdim == 2 and rdim == 3:
                data.inv_grad = np.zeros((2, 3, data.grad.shape[2], data.grad.shape[3]))
            else:
                raise RuntimeError("Unable to calculate inv_grad for tdim={} and rdim={}.".format(self.fe.tdim, rdim))
        # todo: conormal
        return data
    

# class Expression
    

class assembler:

    def __init__(self, 
                 test_space: FiniteElement, 
                 trial_space: Optional[FiniteElement], 
                 quadOrder: int) -> None:
        self.test_space = test_space
        self.trial_space = trial_space
        self.quadOrder = quadOrder
        if trial_space is not None:
            assert(test_space.mesh is trial_space.mesh)

    def assembleFunctional(self, norm, mea: Measure, **kwargs) -> float:
        return 0.0

    def assembleLinear(self, form, mea: Measure, **kwargs) -> np.ndarray:
        # assert(mea.tdim <= self.mesh.tdim)
        # basis_type = self.test_space
        # for _ in range(self.mesh.tdim - mea.tdim):
        #     basis_type = basis_type.trace_type
        pass
    
    def assembleBilinear(self, form, mea: Measure, **kwargs) -> csr_matrix:
        raise NotImplementedError
    
def setMeshMapping(mesh: Mesh, mapping: Optional[Function] = None):
    if mapping is None:
        # set an affine mapping
        raise NotImplementedError
    else:
        mesh.mapping = mapping
    