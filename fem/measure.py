from typing import Union, Optional
import numpy as np
from .mesh import Mesh

class Measure:
    
    mesh: Mesh
    elem_ix: Union[np.ndarray, slice] # the element indices involved


class CellMeasure(Measure):
    """
    Represent the volume measure whose dimension equals the topological dimension of the mesh. 
    """

    # mesh: Mesh
    # elem_ix: Union[np.ndarray, slice]

    def __init__(self, mesh: Mesh, tags: Optional[tuple[int]] = None) -> None:
        self.mesh = mesh
        if tags is None:
            self.elem_ix = slice(None) # select all the elements
        else:
            # select the elements with the provided tags
            elem_tag = mesh.cell_tag[mesh.tdim]
            flag = np.zeros((elem_tag.shape[0],), dtype=np.bool8)
            for t in tags:
                flag[elem_tag == t] = True
            self.elem_ix = np.nonzero(flag)[0]

class FaceMeasure(Measure):
    """
    Represent the surface measure whose dimension is one less than the topological dimension of the mesh. 
    """
    
    # mesh: Mesh
    facet_ix: tuple[np.ndarray]
    # elem_ix: tuple[np.ndarray]
    facet_id: tuple[np.ndarray]

    def __init__(self, mesh: Mesh, tags: Optional[tuple[int]] = None, interiorFacet: bool = False) -> None:
        self.mesh = mesh
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
