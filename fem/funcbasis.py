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
    data: tuple[QuadData] # size = number of local dofs


    def __init__(self, fs: FunctionSpace, mea: Measure) -> None:
        self.fs = fs
        self.mea = mea
        self.update()

    def update(self) -> None:
        mea = self.mea
        Nq = self.mea.quad_w.size
        tdim, rdim = self.fs.elem.tdim, self.fs.elem.rdim
        num_local_dof = self.fs.elem.num_local_dof
        assert mea.mesh is self.fs.mesh
        self.data = []
        #
        if mea.dim == tdim:
            for i in range(num_local_dof):
                u, du = self.fs.elem._eval(i, self.mea.quad_tab.T)
                # u: (rdim, Nq)
                # du: (rdim, tdim, Nq)
                data = QuadData(u.reshape(rdim, 1, Nq)) # (rdim, 1, Nq)
                data.grad = np.einsum("ij...,jk...->ik...", \
                                      du[:,:,np.newaxis,:], mea.x.inv_grad) # (rdim, tdim, Ne, Nq)
                self.data.append(data)
        #
        elif mea.dim == tdim-1:
            for i in range(num_local_dof):
                u, du = self.fs.elem._eval(i, self.mea.quad_tab.reshape(-1, tdim).T)
                # u: (rdim, Nf * Nq)
                # du: (rdim, tdim, Nf * Nq)
                # facet_id: (Nf*, )
                data = QuadData(u.reshape(rdim, -1, Nq)) # (rdim, Nf*, Nq)
                du = du.reshape(rdim, tdim, -1, Nq) # (rdim, tdim, Nf*, Nq)
                data.grad = np.einsum("ij...,jk...->ik...", du, mea.x.inv_grad)
                self.data.append(data)
        #
        else:
            raise RuntimeError("Incorrect measure dimension. ")
        # transform the data into a tuple
        self.data = tuple(self.data)

