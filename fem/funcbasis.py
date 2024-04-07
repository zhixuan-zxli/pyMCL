from typing import Union
import numpy as np
from .function import FunctionSpace, QuadData
from .measure import Measure

class FunctionBasis:
    """
    Generate the quadrature data for each local basis, 
    given a finite element space and the integral domain. 
    """

    fs: FunctionSpace
    mea: Measure
    data: tuple[Union[QuadData, tuple[QuadData]]] # size = number of local dofs


    def __init__(self, fs: FunctionSpace, mea: Measure) -> None:
        self.fs = fs
        self.mea = mea
        self.update()

    def update(self) -> None:
        tdim, rdim = self.fs.elem.tdim, self.fs.elem.rdim
        num_local_dof = self.fs.elem.num_local_dof
        assert self.mea.mesh is self.fs.mesh
        quad_tab = self.mea.quad_tab
        Nq = quad_tab.shape[1]
        self.data = []
        #
        if self.mea.dim == tdim:
            for i in range(num_local_dof):
                u, du = self.fs.elem._eval(i, quad_tab)
                # u: (rdim, Nq)
                # du: (rdim, tdim, Nq)
                data = QuadData(u.reshape(rdim, 1, Nq)) # (rdim, 1, Nq)
                data.grad = np.einsum("ij...,jk...->ik...", \
                                      du[:,:,np.newaxis,:], self.mea.x.inv_grad) # (rdim, tdim, Ne, Nq)
                self.data.append(data)
        #
        elif self.mea.dim == tdim-1:
            quad_tab = self.fs.elem.ref_cell._broadcast_facet(quad_tab) # (tdim, num_facet, Nq)
            for i in range(num_local_dof):
                data_tuple = []
                for y, facet_id in zip(self.mea.x, self.mea.facet_id):
                    u, du = self.fs.elem._eval(i, quad_tab.reshape(tdim, -1))
                    # u: (rdim, num_facet * Nq)
                    # du: (rdim, tdim, num_facet* Nq)
                    # facet_id: (Ne, )
                    data = QuadData(u.reshape(rdim, -1, Nq)[:, facet_id, :]) # (rdim, Ne, Nq)
                    du = du.reshape(rdim, tdim, -1, Nq)[:,:,facet_id,:] # (rdim, tdim, Ne, Nq)
                    data.grad = np.einsum("ij...,jk...->ik...", \
                                          du, y.inv_grad)
                    data_tuple.append(data)
                self.data.append(tuple(data_tuple))
        #
        else:
            raise RuntimeError("Incorrect measure dimension. ")
        # transform the data into a tuple
        self.data = tuple(self.data)

