from typing import Union
import numpy as np
from .function import FunctionSpace, QuadData
from .measure import Measure, CellMeasure, FaceMeasure

class FunctionBasis:
    """
    Generate the quadrature data for each local basis, 
    given a finite element space and the integral domain. 
    """

    fs: FunctionSpace
    mea: Measure
    data: tuple[Union[QuadData, tuple[QuadData]]] # size = number of local dofs

    def __init__(self, fs: FunctionSpace, mea: Measure, quad_tab: np.ndarray, x: QuadData) -> None:
        self.fs = fs
        self.mea = mea
        tdim, rdim = fs.elem.tdim, fs.elem.rdim
        num_local_dof = fs.elem.num_local_dof
        assert mea.mesh is fs.mesh
        Nq = quad_tab.shape[1]
        self.data = []
        #
        if isinstance(mea, CellMeasure):
            for i in range(num_local_dof):
                u, du = fs.elem._eval(i, quad_tab)
                # u: (rdim, Nq)
                # du: (rdim, tdim, Nq)
                data = QuadData(u.reshape(rdim, 1, Nq)) # (rdim, 1, Nq)
                data.grad = np.einsum("ij...,jk...->ik...", \
                                      du[:,:,np.newaxis,:], x.inv_grad) # (rdim, tdim, Ne, Nq)
                self.data.append(data)
        #
        elif isinstance(mea, FaceMeasure):
            quad_tab = fs.elem.ref_cell._broadcast_facet(quad_tab) # (tdim, num_facet, Nq)
            for i in range(num_local_dof):
                data_tuple = []
                for k, facet_id in enumerate(mea.facet_id):
                    u, du = fs.elem._eval(i, quad_tab.reshape(tdim, -1))
                    # u: (rdim, num_facet * Nq)
                    # du: (rdim, tdim, num_facet* Nq)
                    # facet_id: (Ne, )
                    data = QuadData(u.reshape(rdim, -1, Nq)[:, facet_id, :]) # (rdim, Ne, Nq)
                    du = du.reshape(rdim, tdim, -1, Nq)[:,:,facet_id,:] # (rdim, tdim, Ne, Nq)
                    data.grad = np.einsum("ij...,jk...->ik...", \
                                          du, x[k].inv_grad)
                    data_tuple.append(data)
                self.data.append(tuple(data_tuple))
        #
        self.data = tuple(self.data)

