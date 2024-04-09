from typing import Optional
import numpy as np
from scipy.sparse import coo_array
from .mesh import Mesh
from .element import Element
from .refdom import ref_doms
from tools.binsearchkw import binsearchkw

class FunctionSpace:

    mesh: Mesh
    elem: Element
    periodic: bool
    
    dof_loc: np.ndarray # (num_dof, gdim)
    dof_group: dict[str, np.ndarray]
    elem_dof: np.ndarray
    # elem_dof: (num_local_dof, Ne)
    # where num_local_dof[d] = num_dof_loc[d] * num_sub_ent[d] * num_dof_type[d], 
    # and d = 0, 1, ..., tdim
    facet_dof: np.ndarray
    # layout same as above
    num_dof: int


    def __init__(self, mesh: Mesh, elem: Element, periodic: bool = False) -> None:
        self.mesh = mesh
        self.elem = elem
        self.periodic = periodic

        tdim = elem.tdim
        assert mesh.tdim == tdim
        elem_cell = mesh.cell[tdim]
        num_elem = elem_cell.shape[0]
        facet_cell = mesh.cell[tdim-1]
        num_facet = facet_cell.shape[0]

        self.dof_loc = np.zeros((0, mesh.gdim))
        self.dof_group = dict()
        self.elem_dof = np.zeros((0, num_elem), dtype=np.int32)
        self.facet_dof = np.zeros((0, num_facet), dtype=np.int32)

        # Build the dof for entities of each dimension. 
        offset = 0
        for d in range(tdim+1):
            if elem.dof_loc[d] is None:
                assert elem.dof_name[d] is None
                continue

            # 1. calculate the dof locations
            num_dof_loc = elem.dof_loc[d].shape[0]
            sub_ent = elem.ref_cell._get_sub_entities(elem_cell, dim=d) # (Ne, num_sub_ent, d+1)
            num_sub_ent = num_elem * sub_ent.shape[1]
            rows = np.broadcast_to(np.arange(num_sub_ent, dtype=np.int32)[:,np.newaxis], (num_sub_ent, d+1))
            vals = np.zeros((num_sub_ent, d+1))
            coo = coo_array((vals.reshape(-1), (rows.reshape(-1), sub_ent.reshape(-1))), \
                            shape=(num_sub_ent, mesh.point.shape[0]))
            all_locs = np.zeros((0, mesh.gdim)) # eventually (num_dof_loc * num_total_sub_ent, gdim)
            for loc in elem.dof_loc[d]:
                coo.data = np.broadcast_to(loc[np.newaxis], (num_sub_ent, d+1)).reshape(-1)
                all_locs = np.vstack((all_locs, coo @ mesh.point))

            # match the dof locations
            _, fw_idx, inv_idx = np.unique(all_locs.round(decimals=10), return_index=True, return_inverse=True, axis=0)
            # the magic number 10 here may not be robust enough ^
            uq_locs = all_locs[fw_idx] # to get rid off the rounding effect
            inv_idx = inv_idx.astype(np.int32)
            # inv_idx: (num_dof_loc * num_total_sub_ent, )

            # 2. Broadcast to multiple dof types; save the dof locations
            num_new_dof = uq_locs.shape[0]
            num_dof_type = len(elem.dof_name[d])
            self.dof_loc = np.vstack((self.dof_loc, np.repeat(uq_locs, repeats=num_dof_type, axis=0)))
            for i, name in enumerate(elem.dof_name[d]):
                new_dof_idx = offset + np.arange(num_new_dof, dtype=np.int32) * num_dof_type + i
                if name not in self.dof_group:
                    self.dof_group[name] = new_dof_idx
                else:
                    self.dof_group[name] = np.concatenate((self.dof_group[name], new_dof_idx))

            # 3. Save the dof table for each element. 
            new_elem_dof = inv_idx.reshape(num_dof_loc, num_elem, -1).transpose(0, 2, 1)[:,:,np.newaxis,:] # (num_dof_loc, num_sub_ent, 1, num_elem)
            new_elem_dof = offset + new_elem_dof * num_dof_type + np.arange(num_dof_type).reshape(1, 1, -1, 1) # (num_dof_loc, num_sub_ent, num_dof_type, num_elem)
            self.elem_dof = np.vstack((self.elem_dof, new_elem_dof.reshape(-1, num_elem))) 
            
            # 3. Find the dof number for the facet dofs. 
            if d < tdim and num_facet > 0:
                sub_ent = ref_doms[tdim-1]._get_sub_entities(facet_cell, dim=d) # (Nf, num_sub_ent, d+1)
                num_sub_ent = num_facet * sub_ent.shape[1]
                rows = np.broadcast_to(np.arange(num_sub_ent, dtype=np.int32)[:,np.newaxis], (num_sub_ent, d+1))
                vals = np.zeros((num_sub_ent, d+1))
                coo = coo_array((vals.reshape(-1), (rows.reshape(-1), sub_ent.reshape(-1))), \
                                shape=(num_sub_ent, mesh.point.shape[0]))
                all_locs = np.zeros((0, mesh.gdim)) # eventually (num_dof_loc * num_sub_ent, gdim)
                for loc in elem.dof_loc[d]:
                    coo.data = np.broadcast_to(loc[np.newaxis], (num_sub_ent, d+1)).reshape(-1)
                    all_locs = np.vstack((all_locs, coo @ mesh.point))
                f_idx = binsearchkw(uq_locs.round(decimals=10), all_locs.round(decimals=10)) # (num_dof_loc * num_sub_ent, )
                # the magic number 10 here may not be robust enough ^
                assert np.all(f_idx != -1)
                # save to facet dof
                new_facet_dof = f_idx.reshape(num_dof_loc, num_facet, -1).transpose(0, 2, 1)[:,:,np.newaxis,:] # (num_dof_loc, num_sub_ent, 1, num_facet)
                new_facet_dof = offset + new_facet_dof * num_dof_type + np.arange(num_dof_type).reshape(1, 1, -1, 1) # (num_dof_loc, num_sub_ent, num_dof_type, num_facet)
                self.facet_dof = np.vstack((self.facet_dof, new_facet_dof.reshape(-1, num_facet)))

            # end for dof_name
            offset += num_new_dof * num_dof_type
        # end for d
        assert elem.num_local_dof == self.elem_dof.shape[0]
        assert offset == self.dof_loc.shape[0]
        self.num_dof = offset

    def getFacetDof(self, tags: Optional[tuple[int]] = None) -> np.ndarray:
        if tags is None:
            flag = slice(None)
        else:
            facet_tag = self.mesh.cell_tag[self.mesh.tdim-1]
            flag = np.zeros((facet_tag.shape[0], ), dtype=np.bool8)
            for t in tags:
                flag[facet_tag == t] = True
        return self.facet_dof[:, flag]
    
# def group_dof(mixed_fe: tuple[Element], dof_list) -> np.ndarray:
#     """
#     Get the free dof (as a bool mask) for a mixed finite element space. 
#     mixed_fe: a tuple of Element objects. 
#     dof_list: a tuple of dof lists. dof_list[c] is the fixed dof for mixed_fe[c]. 
#     It can be 
#     (1) None, or a tuple of array of dofs to be fixed if mixed_fe[c] has a single copy, 
#     (2) or a tuple of (1) for each copy of the finite element space. 
#     """
#     total_num_dof = sum(fe.num_dof * fe.num_copy for fe in mixed_fe)
#     free_dof = np.ones((total_num_dof, ), dtype=np.bool8)
#     # combine these dof to get the free dof
#     base_index = 0
#     for fe, dof in zip(mixed_fe, dof_list):
#         if fe.num_copy == 1:
#             dof = (dof, )
#         for c in range(fe.num_copy):
#             if dof[c] is None:
#                 continue
#             assert isinstance(dof[c], tuple)
#             for dd in dof[c]:
#                 free_dof[dd.reshape(-1)*fe.num_copy+c+base_index] = False
#         base_index += fe.num_dof * fe.num_copy
#     return free_dof