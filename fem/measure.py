from typing import Any, Union, Optional
import numpy as np
from .mesh import Mesh
from .refdom import ref_doms
from .quadrature import Quadrature
from .funcspace import _calculate_dof_locations

class Measure:
    
    mesh: Mesh
    dim: int
    doubleSided: bool
    elem_ix: Union[np.ndarray, slice] # the element indices involved
    facet_ix: np.ndarray # the facet indices of a surface measure
    facet_id: np.ndarray # the facet if within an element, for a surface measure

    quad_tab: np.ndarray # (num_quad, tdim) if volume measure, or (num_facets, num_quad, tdim) if surface measure
    quad_w: np.ndarray # (num_quad, )
    x: Any # type: QuadData # geometric quantities provided
    

    def __init__(self, 
                 mesh: Mesh, 
                 dim: int, 
                 order: int, 
                 tags: Optional[tuple[int]] = None, 
                 doubleSided: bool = False, 
                 coord_map = None) -> None:
        """
        If dim == mesh.tdim, represent the volume measure. 
        If dim == mesh.tdim-1, represent the surface measure. 
        """
        self.mesh = mesh
        self.dim = dim
        self.doubleSided = doubleSided
        self.quad_tab, self.quad_w = Quadrature.getTable(ref_doms[dim], order)
        # 1. Volume measure
        if dim == mesh.tdim:
            # prepare the element indices
            if tags is None:
                self.elem_ix = slice(None) # select all the elements
            else:
                # select the elements with the provided tags
                elem_tag = mesh.cell_tag[mesh.tdim]
                flag = np.zeros((elem_tag.shape[0],), dtype=np.bool_)
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
                flag = np.zeros((facet_tag.shape[0], ), dtype=np.bool_)
                for t in tags:
                    flag[facet_tag == t] = True
                self.facet_ix = np.nonzero(flag)[0]
            #
            if doubleSided:
                assert np.all(mesh.facet_ref[0,0,self.facet_ix] != mesh.facet_ref[1,0,self.facet_ix]), \
                "The selected facets contain boundary facets. "
            n = 1 + doubleSided
            self.elem_ix = mesh.facet_ref[:n, 0, self.facet_ix].reshape(-1) # (n * num_facets, )
            self.facet_id = mesh.facet_ref[:n, 1, self.facet_ix].reshape(-1) # (n * num_facets, )
            # calculate the quadrature locations
            facet_entities = ref_doms[dim]._get_sub_entities(mesh.cell[dim][self.facet_ix], dim=dim) # (num_facets, 1, dim+1)
            quad_locs = _calculate_dof_locations(mesh, facet_entities, self.quad_tab) # (num_quad, num_facets, 1, gdim)
            quad_locs = quad_locs.reshape(self.quad_w.size, -1, mesh.gdim) # (num_quad, num_facets, gdim)
            quad_locs = np.tile(quad_locs, (1, n, 1)) # (num_quad, n*num_facets, gdim) # Why tile for interior facets?
            # transform to local barycentric coordinates
            self.quad_tab = self._global_to_local(quad_locs) # (n*num_facets, num_quad, tdim)
        else:
            raise RuntimeError("This measure is neither a volume measure nor a surface measure.")
        #
        self.update(coord_map)

    def _global_to_local(self, quad_locs: np.ndarray) -> np.ndarray:
        """
        Transform the global coordinates to local coordinates within elements. 
        quad_locs: (num_quad, num_facets, gdim)
        return: (num_facets, num_quad, tdim)
        """
        tdim = self.mesh.tdim
        num_quad, num_facets = quad_locs.shape[:2]
        local_x = np.zeros((num_facets, num_quad, tdim))
        elem_cell = self.mesh.cell[tdim]
        verts = tuple(self.mesh.point[elem_cell[self.elem_ix, j]] for j in range(tdim+1)) # (tdim+1, num_facets, gdim)
        if tdim == 1:
            denom = np.linalg.norm(verts[1] - verts[0], axis=1)
            for i, loc in enumerate(quad_locs): # loc: (num_facets, gdim)
                local_x[:,i,0] = np.linalg.norm(loc - verts[0], axis=1) / denom
        elif tdim == 2:
            def _a(q0, q1, q2):
                c = np.cross(q1-q0, q2-q0, axis=1)
                return np.linalg.norm(c, axis=1) if c.ndim == 2 else c
            denom = _a(verts[0], verts[1], verts[2])
            for i, loc in enumerate(quad_locs):
                local_x[:,i,0] = _a(verts[0], loc, verts[2]) / denom
                local_x[:,i,1] = _a(verts[0], verts[1], loc) / denom
        else:
            raise NotImplementedError
        assert local_x.min() > -1e-3 and local_x.sum(axis=2).max() < 1.0+1e-3
        return local_x


    def update(self, coord_map = None) -> None:
        """
        Update the quadrature data for this measure, 
        including the coordinates, the transformation gradient and its inverse, and the cell normal. 
        If the measure is a surface one, 
        update also the facet normal and the surface Jacobian. 
        """
        self.x = None
        if coord_map is None:
            coord_map = self.mesh.coord_map
        temp = coord_map._interpolate(self)
        self.x = temp
        self._derive_geometric_quantities(self.x)
        # Update the quadrature data for a surface measure. 
        if self.dim == self.mesh.tdim - 1:
            ref_fn = ref_doms[self.mesh.tdim].facet_normal.T # (tdim, num_facet)
            # derive face normal, face measure: 
            # given inv_grad: (tdim, rdim, num_facet, num_quad)
            # build fn: (rdim, num_facet, num_quad)
            # build ds: (1, num_facet, num_quad)
            self.x.fn = np.sum(self.x.inv_grad * ref_fn[:, np.newaxis, self.facet_id, np.newaxis], axis=0) # (rdim, num_facet*n, num_quad)
            nm = np.linalg.norm(self.x.fn, ord=None, axis=0, keepdims=True) # (1, num_facet*n, num_quad)
            self.x.fn = self.x.fn / nm
            self.x.ds = self.x.dx * nm

    def _derive_geometric_quantities(self, data) -> None:
        """
        Given data of type QuadData of a mesh mapping, 
        derive the geometric quantities like cell measure, cell normal and gradient inverse. 
        """
        Ne, Nq = data.shape[1:] # Ne = num_elem if volume measure, or n * num_facets if surface measure
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
                data.cn = data.cn / data.dx[np.newaxis]
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
