import numpy as np
from scipy.sparse import csr_matrix
from mesh import Mesh

class Element:
    def __init__(self, type: str, degree: int) -> None:
        self.type = type
        self.degree = degree

    def __str__(self) -> str:
        return "Degree-{} {} element".format(self.degree, self.type)

class FESpace:
    def __init__(self, mesh: Mesh, elem: Element, vecdim: int = 1) -> None:
        self.mesh = mesh
        self.elem = elem
        self.vecdim = vecdim
        initializer = {"Lagrange": self.__init_Lagrange}
        if not elem.type in initializer:
            raise RuntimeError("Unrecognized finite element type {}. ".format(elem.type))
        initializer[elem.type](elem.degree)
        # todo: handle periodicity here

    def __init_Lagrange(self, degree: int) -> None:
        if degree >= 1:
            if hasattr(self.mesh, "tri"):
                self.dof_of_elem = self.mesh.tri[:,:-1]
            elif hasattr(self.mesh, "edge"):
                self.dof_of_elem = self.mesh.edge[:,:-1]
            else:
                self.dof_of_elem = np.arange(self.mesh.point.shape[0]).reshape(-1, 1)
        if degree == 2:
            if hasattr(self.mesh, "tri"):
                # figure out the new dofs on the edge
                Np = self.mesh.point.shape[0]
                Nt = self.dof_of_elem.shape[0]
                tri_edges = self.dof_of_elem[:, [0,1,1,2,2,0]].reshape(-1, 3, 2)
                S = csr_matrix((np.ones((3*Nt,), dtype=np.uint32), 
                            (np.min(tri_edges, axis=2).flatten(), np.max(tri_edges, axis=2).flatten())), 
                            shape=(Np, Np))
                S.data = np.arange(S.nnz, dtype=S.dtype) + (Np+1)
                extra_dof = S[np.min(tri_edges, axis=2), np.max(tri_edges, axis=2)].todense()
            elif hasattr(self.mesh, "edge"):
                extra_dof = np.arange(self.edge.shape[0]).reshape(-1, 1) + self.mesh.point.shape[0]
            self.dof_of_elem = np.hstack((self.dof_of_elem, extra_dof))
        if degree >= 3:
            raise RuntimeError("Degree {} not implemented for Lagrange element. ".format(degree))
