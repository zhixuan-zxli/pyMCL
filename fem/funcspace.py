from typing import Optional
from warnings import warn
import numpy as np
from .mesh import Mesh
from .refdom import ref_doms
from .element import Element, VectorElement
from tools.binsearchkw import binsearchkw

class FunctionSpace:
    """
    This function accepts a mesh and a finite element type and generates the DOFs for the function space.
    """
    mesh: Mesh
    elem: Element
    
    dof_loc: np.ndarray # (num_dof, gdim)
    elem_dof: np.ndarray
    # of shape (num_local_dof, Ne)
    # where num_local_dof = \sum_{d=0}^{tdim} num_sub_ent[d] * num_dof_loc[d] * num_dof_type[d], 
    # and d = 0, 1, ..., tdim
    
    num_dof: int

    def __init__(self, mesh: Mesh, elem: Element, constraint: Optional[callable] = None) -> None:
        self.mesh = mesh
        self.elem = elem
        tdim, gdim = elem.tdim, mesh.point.shape[-1]
        assert mesh.tdim == tdim
        elem_cell = mesh.cell[tdim]
        num_elem = elem_cell.shape[0]

        self.dof_loc = np.zeros((0, mesh.gdim))
        self.elem_dof = np.zeros((0, num_elem), dtype=np.int32)

        # Build the dof for entities of each dimension. 
        offset = 0
        for d in range(tdim+1):
            elem_dof_loc = elem.dof_loc[d]
            if elem_dof_loc is None:
                assert elem.dof_name[d] is None
                continue
            # 1. calculate and match the dof locations. 
            num_dof_loc = elem_dof_loc.shape[0]
            all_locs = _calculate_dof_locations(mesh, ref_doms[tdim]._get_sub_entities(elem_cell, dim=d), elem_dof_loc)
            all_locs = all_locs.reshape(-1, gdim)
            # of shape (num_dof_loc * num_elem * num_sub_entities, gdim)
            
            if not elem.discontinuous: 
                if constraint is not None:
                    constraint(all_locs)
                _, fw_idx, inv_idx = np.unique(all_locs.round(decimals=10), return_index=True, return_inverse=True, axis=0)
                # the magic number 10 here may not be robust enough ^
                uq_locs = all_locs[fw_idx] # to get rid off the rounding effect
                inv_idx = inv_idx.astype(np.int32)
            else: 
                if constraint is not None:
                    warn("The constraint is not supported for discontinuous elements. ")
                fw_idx = np.lexsort(all_locs[:, ::-1].T)
                uq_locs = all_locs[fw_idx, :]
                inv_idx = np.empty_like(fw_idx, dtype=np.int32)
                inv_idx[fw_idx] = np.arange(all_locs.shape[0], dtype=np.int32)
            # inv_idx: (num_dof_loc * num_elem * num_sub_entities, )

            # 2. Broadcast to multiple dof types; save the dof locations and the dof table. 
            num_new_dof = uq_locs.shape[0]
            num_dof_name = len(elem.dof_name[d])
            self.dof_loc = np.vstack((self.dof_loc, np.repeat(uq_locs, repeats=num_dof_name, axis=0)))
            new_elem_dof = inv_idx.reshape(num_dof_loc, num_elem, -1).transpose(2, 0, 1)[:,:,np.newaxis,:] # of shape (num_sub_entites, num_dof_loc, 1, num_elem)
            new_elem_dof = offset + new_elem_dof * num_dof_name + np.arange(num_dof_name).reshape(1, 1, -1, 1) # (num_sub_ent, num_dof_loc, num_dof_type, num_elem)
            self.elem_dof = np.vstack((self.elem_dof, new_elem_dof.reshape(-1, num_elem))) 
            
            offset += num_new_dof * num_dof_name
        # end for d
        assert elem.num_local_dof == self.elem_dof.shape[0]
        assert offset == self.dof_loc.shape[0]
        self.num_dof = offset

    def getDofByLocation(self, loc: np.ndarray, round_dec: int = 10) -> np.ndarray:
        """
        Get the dof indices by the dof locations. 
        loc: (n, gdim), without repeating locations for a vector element. 
        return: (n * nc, ), where nc is the number of componenets for a vector element. 
        """
        nc = 1 if not isinstance(self.elem, VectorElement) else self.elem.rdim
        # get the cached sorted dof locations
        if not hasattr(self, '_sorted_dof_loc'):
            self._sorted_indices = np.lexsort(self.dof_loc[::nc, ::-1].T)
            self._sorted_dof_loc = self.dof_loc[self._sorted_indices*nc]
        dof = binsearchkw(self._sorted_dof_loc.round(decimals=round_dec), loc.round(decimals=round_dec))
        assert np.all(dof != -1), "Some dof locations are not found. "
        dof = self._sorted_indices[dof] * nc
        if nc > 1:
            dof = np.repeat(dof, nc)
            for d in range(nc):
                dof[d::nc] += d
        return dof
    
def _calculate_dof_locations(mesh: Mesh, sub_ent: np.ndarray, dof_loc: np.ndarray) -> np.ndarray:
    # sub_ent: (num_elem, num_sub_entities, tdim+1)
    # dof_loc: (num_dof_loc, tdim+1)
    # result: (num_dof_loc, num_elem, num_sub_entities, gdim)
    tdim = sub_ent.shape[-1] - 1 # the dimension of the sub entities
    gdim = mesh.point.shape[-1] # the geometric dimension
    result = np.zeros((dof_loc.shape[0], sub_ent.shape[0], sub_ent.shape[1], gdim))
    for loc, r in zip(dof_loc, result):
        for i in range(tdim+1):
            r += mesh.point[sub_ent[:,:,i], :] * loc[i] # of shape (num_elem, num_sub_entities, gdim)
    return result


def group_dof(mixed_fe: tuple[FunctionSpace], dof_list: tuple[Optional[np.ndarray]]) -> np.ndarray:
    """
    Get the free dof (as a bool mask) for a mixed finite element space. 
    mixed_fe: a tuple of Element objects. 
    dof_list: a tuple of dof lists. dof_list[c] is either None or a sequence of fixed dof indices. 
    """
    total_num_dof = sum(fe.num_dof for fe in mixed_fe)
    free_dof = np.ones((total_num_dof, ), dtype=np.bool8)
    # combine these dof to get the free dof
    offset = 0
    for fs, dof in zip(mixed_fe, dof_list):
        if dof is not None:
            free_dof[offset+dof] = False
        offset += fs.num_dof
    return free_dof
