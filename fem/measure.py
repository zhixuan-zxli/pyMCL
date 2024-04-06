from typing import Union, Optional
import numpy as np
from .mesh import Mesh

class Measure:
    
    mesh: Mesh
    dim: int
    elem_ix: Union[np.ndarray, slice] # the element indices involved
    facet_ix: tuple[np.ndarray] # the facet indices of a surface measure
    facet_id: tuple[np.ndarray] # the facet if within an element, for a surface measure

    # todo: move MeshMapping._interpolate here ...

    def __init__(self, 
                 mesh: Mesh, 
                 dim: int, 
                 tags: Optional[tuple[int]] = None, 
                 interiorFacet: bool = False) -> None:
        """
        If dim == mesh.tdim, represent the volume measure. 
        If dim == mesh.tdim-1, represent the surface measure. 
        """
        self.mesh = mesh
        self.dim = dim
        # 1. Volume measure
        if dim == mesh.tdim:
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
            for k in range(1+interiorFacet):
                elem_ix.append(mesh.facet_ref[k,0,self.facet_ix])
                facet_id.append(mesh.facet_ref[k,1,self.facet_ix])
            self.elem_ix = tuple(elem_ix)
            self.facet_id = tuple(facet_id)
        else:
            raise RuntimeError("This measure is neither a volume measure nor a surface measure.")
