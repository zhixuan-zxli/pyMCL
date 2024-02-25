from typing import List, Optional
import numpy as np
from scipy.sparse import csr_matrix
from mesh import Mesh
import element

class FiniteElement:
    def __init__(self, mesh: Mesh, elem: element.Element, vecdim: int = 1) -> None:
        self.mesh = mesh
        self.elem = elem
        self.vecdim = vecdim
        initializer = {element.LagrangeTri: self._init_LagrangeTri}
        initializer[type(elem)](elem.degree)
        # todo: handle periodicity here

    def _init_LagrangeTri(self, degree: int) -> None:
        if degree >= 1:
            self.cell_dof = self.mesh.tri[:, :-1]
            self.facet_dof = self.mesh.edge[:, :-1]
            self.num_dof = self.mesh.point.shape[0]
        if degree == 2:
            Np = self.mesh.point.shape[0]
            # figure out the new dofs on the edge
            Nt = self.cell_dof.shape[0]
            tri_edges = self.mesh.tri[:, [0,1,1,2,2,0]].reshape(-1, 3, 2)
            S = csr_matrix((np.ones((3*Nt,), dtype=np.uint32), 
                        (np.min(tri_edges, axis=2).flatten(), np.max(tri_edges, axis=2).flatten())), 
                        shape=(Np, Np))
            assert(S.nnz == Np + Nt - 1) # check using Euler's formula
            S.data = np.arange(S.nnz, dtype=S.dtype) + Np
            # build dof for elements
            extra_dof = S[np.min(tri_edges, axis=2).flatten(), np.max(tri_edges, axis=2).flatten()]
            extra_dof = extra_dof.reshape(-1, 3)
            self.cell_dof = np.hstack((self.cell_dof, extra_dof))
            # build dof for facets
            edge = self.mesh.edge[:, :-1]
            extra_dof = S[np.min(edge, axis=1), np.max(edge, axis=1)]
            extra_dof = extra_dof.reshape(-1, 1)
            self.facet_dof = np.hstack((self.facet_dof, extra_dof))
            self.num_dof = Np + S.nnz
        if degree >= 3:
            raise RuntimeError("Degree {} not implemented for {}".format(degree, self.elem))
        
    def getFacetDof(self, ids: Optional[List[int]] = None) -> np.ndarray:
        if self.elem.tdim == 3:
            raise RuntimeError("getFacetDof for 3D is not implemented. ")
        if self.elem.tdim == 2:
            if ids is None:
                return np.unique(self.facet_dof)
            facet_flag = np.zeros((self.mesh.edge.shape[0], ), dtype=np.bool8)
            for i in ids:
                facet_flag[self.mesh.edge[:, -1] == i] = True
            return np.unique(self.facet_dof[facet_flag, :])
        if self.elem.tdim == 1:
            if ids is None:
                return np.arange(self.facet_dof)
            point_tag = self.mesh.point[self.mesh.point[:, -1] > 0, -1]
            facet_flag = np.zeros((point_tag.shape[0], ), np.bool8)
            for i in ids:
                facet_flag[point_tag == i] = True
            return np.unique(self.facet_dof[facet_flag])
            
        
        
