from typing import Any, Union, Optional
import numpy as np
from .mesh import Mesh
from .refdom import ref_doms
from .quadrature import Quadrature

class Measure:
    
    mesh: Mesh
    dim: int
    elem_ix: Union[np.ndarray, slice] # the element indices involved
    facet_ix: np.ndarray # the facet indices of a surface measure
    facet_id: np.ndarray # the facet if within an element, for a surface measure

    quad_tab: np.ndarray
    x: Any # type: QuadData # geometric quantities provided
    

    def __init__(self, 
                 mesh: Mesh, 
                 dim: int, 
                 order: int, 
                 tags: Optional[tuple[int]] = None, 
                 interiorFacet: bool = False) -> None:
        """
        If dim == mesh.tdim, represent the volume measure. 
        If dim == mesh.tdim-1, represent the surface measure. 
        """
        self.mesh = mesh
        self.dim = dim
        self.quad_tab = Quadrature.getTable(ref_doms[dim], order)
        # 1. Volume measure
        if dim == mesh.tdim:
            assert interiorFacet == False
            # prepare the element indices
            if tags is None:
                self.elem_ix = slice(None) # select all the elements
            else:
                # select the elements with the provided tags
                elem_tag = mesh.cell_tag[mesh.tdim]
                flag = np.zeros((elem_tag.shape[0],), dtype=np.bool8)
                for t in tags:
                    flag[elem_tag == t] = True
                self.elem_ix = np.nonzero(flag)[0]
        # 2. Surface measure
        elif dim == mesh.tdim-1:
            # prepare the facet/element indices
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
            if interiorFacet:
                assert np.all(mesh.facet_ref[0,0,self.facet_ix] != mesh.facet_ref[1,0,self.facet_ix]), \
                "The selected facets contain boundary facets. "
            n = 1 + interiorFacet
            self.elem_ix = mesh.facet_ref[:n,0,self.facet_ix].reshape(-1) # (n * Nf, )
            self.facet_id = mesh.facet_ref[:n,1,self.facet_ix].reshape(-1) # (n * Nf, )
        else:
            raise RuntimeError("This measure is neither a volume measure nor a surface measure.")
        #
        self.update()


    def update(self) -> None:
        self.x = None
        temp = self.mesh.coord_map._interpolate(self)
        self.x = temp
        self._derive_geometric_quantities(self.x)
        # Update the quadrature data for a surface measure. 
        if self.dim == self.mesh.tdim - 1:
            ref_fn = ref_doms[self.mesh.tdim].facet_normal.T # (tdim, num_facet)
            # derive face normal, face measure: 
            # given inv_grad: (tdim, rdim, Nf, Nq)
            # build fn: (rdim, Nf, Nq)
            # build ds: (1, Nf, Nq)
            self.x.fn = np.sum(self.x.inv_grad * ref_fn[:, np.newaxis, self.facet_id, np.newaxis], axis=0) # (rdim, Nf*, Nq)
            nm = np.linalg.norm(self.x.fn, ord=None, axis=0, keepdims=True) # (1, Ne, Nq)
            self.x.fn = self.x.fn / nm
            self.x.ds = self.x.dx * nm

    def _derive_geometric_quantities(self, data: np.ndarray):
        """
        Given a QuadData of a mesh mapping, 
        derive the geometric quantities like cell measure, cell normal and gradient inverse. 
        """
        Ne, Nq = data.shape[1:]
        # here tdim is the dimension of the cell but not the measure
        tdim, rdim = self.mesh.tdim, self.mesh.gdim 
        assert data.grad.shape[:2] == (rdim, tdim)
        # given grad: (rdim, tdim, Ne, Nq)
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
