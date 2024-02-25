import numpy as np
from scipy.sparse import csr_matrix
from mesh import Mesh
from element import Element

class FiniteElement:
    def __init__(self, mesh: Mesh, elem: Element, vecdim: int = 1) -> None:
        self.mesh = mesh
        self.elem = elem
        self.vecdim = vecdim
        initializer = {"Lagrange": self.__init_Lagrange__}
        if not elem.type in initializer:
            raise RuntimeError("Unrecognized finite element type {}. ".format(elem.type))
        initializer[elem.type](elem.degree)
        # todo: handle periodicity here

    def __init_Lagrange__(self, degree: int) -> None:
        if degree == 1:
            if self.mesh.tri.shape[0] > 0:
                self.dof_of_elem = self.mesh.tri[:,:-1]
            elif self.mesh.edge.shape[0] > 0:
                self.dof_of_elem = self.mesh.edge[:,:-1]
            else:
                self.dof_of_elem = np.arange(self.mesh.point.shape[0], dtype=np.uint32).reshape(-1, 1)
            self.num_dof = self.mesh.point.shape[0]
        if degree == 2:
            Np = self.mesh.point.shape[0]
            if self.mesh.tri.shape[0] > 0:
                # figure out the new dofs on the edge
                Nt = self.dof_of_elem.shape[0]
                tri_edges = self.dof_of_elem[:, [0,1,1,2,2,0]].reshape(-1, 3, 2)
                S = csr_matrix((np.ones((3*Nt,), dtype=np.uint32), 
                            (np.min(tri_edges, axis=2).flatten(), np.max(tri_edges, axis=2).flatten())), 
                            shape=(Np, Np))
                assert(S.nnz == Np + Nt - 1) # check using Euler's formula
                S.data = np.arange(S.nnz, dtype=S.dtype) + Np
                extra_dof = S[np.min(tri_edges, axis=2).flatten(), np.max(tri_edges, axis=2).flatten()]
                extra_dof = extra_dof.reshape(-1, 3)
                self.num_dof = Np + S.nnz
            elif self.mesh.edge.shape[0] > 0:
                extra_dof = np.arange(self.mesh.edge.shape[0], dtype=np.uint32).reshape(-1, 1) + self.mesh.point.shape[0]
                self.num_dof = Np + self.mesh.edge.shape[0]
            self.dof_of_elem = np.hstack((self.dof_of_elem, extra_dof))
        if degree >= 3:
            raise RuntimeError("Degree {} not implemented for Lagrange element. ".format(degree))
        
