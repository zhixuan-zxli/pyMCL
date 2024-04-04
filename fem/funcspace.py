import numpy as np
from scipy.sparse import csr_array
from .mesh import Mesh
from .element import Element

class FunctionSpace:

    mesh: Mesh
    elem: Element
    periodic: bool
    
    dof_loc: np.ndarray # (num_dof, gdim)
    dof_group: dict[str, np.ndarray]
    elem_dof: np.ndarray
    num_dof: int


    def __init__(self, mesh: Mesh, elem: Element, periodic: bool = False) -> None:
        self.mesh = mesh
        self.elem = elem
        self.periodic = periodic

        num_elem = mesh.cell[tdim].shape[0]
        tdim = elem.tdim
        assert mesh.tdim == tdim

        self.dof_loc = np.zeros((0, mesh.gdim))
        self.dof_group = dict()
        self.elem_dof = np.zeros((0, num_elem), dtype=np.int32)

        # 1. build the dof for entities of each dimension
        offset = 0
        for d in range(tdim):
            if elem.dof_loc[d] is None:
                assert elem.dof_name[d] is None
                continue
            # 2. calculate the dof locations
            sub_ent = elem.ref_cell.sub_entities[d] # (num_sub_ent, d+1)
            num_total_sub_ent = num_elem * sub_ent.shape[0]
            num_dof_loc = elem.dof_loc[d].shape[0]
            all_locs = np.zeros((0, mesh.gdim))
            for loc in elem.dof_loc[d]:
                cols = mesh.cell[tdim][:, sub_ent.ravel()].reshape(num_total_sub_ent, d+1)
                rows = np.tile(np.arange(num_total_sub_ent), (1, d+1))
                vals = np.tile(loc[np.newaxis], (num_total_sub_ent, 1))
                arr = csr_array((vals.reshape(-1), (rows.reshape(-1), cols.reshape(-1))), \
                                shape=(num_total_sub_ent, mesh.point.shape[0]))
                all_locs = np.vstack((all_locs, arr @ mesh.point))
            assert all_locs.shape[0] == num_total_sub_ent * num_dof_loc
            # 3. match the dof locations
            if d == 0:
                uq_locs = mesh.point
                inv_idx = mesh.cell[tdim]
            elif d == tdim:
                uq_locs = all_locs
                inv_idx = np.arange(num_elem)
            else:
                uq_locs, inv_idx = np.unique(all_locs.round(decimals=10), return_index=True, axis=0)
            # the magic number 10 here may not be robust
            # inv_idx: (num_total_sub_ent * num_dof_loc, )
            # 4. broadcast to the dof types and fill in elem_dof
            num_new_dof = uq_locs.shape[0]
            num_dof_type = len(elem.dof_name[d])
            self.dof_loc = np.vstack((self.dof_loc, np.repeat(uq_locs, repeats=num_dof_type, axis=0)))
            for i, name in enumerate(elem.dof_name[d]):
                new_dof_idx = offset + np.arange(num_new_dof) * num_dof_type + i
                if name not in self.dof_group:
                    self.dof_group[name] = new_dof_idx
                else:
                    self.dof_group[name] = np.concatenate((self.dof_group[name], new_dof_idx))
                new_elem_dof = np.zeros((num_dof_type, num_dof_loc, num_elem, sub_ent.shape[0]))
                new_elem_dof[i] = offset + inv_idx.reshape(num_dof_loc, num_elem, -1) * num_dof_type + i
                self.elem_dof = np.vstack((self.elem_dof, new_elem_dof.transpose(3, 1, 0, 2).reshape(-1, num_elem))) 
            #
            offset += num_new_dof * num_dof_type

        # 6. add the element dofs. 
        # if elem.dof_loc[tdim] is not None:
        #     # calculate the dof locations
        #     num_dof_loc = elem.dof_loc[tdim].shape[0]
        #     all_locs = np.zeros((0, mesh.gdim))
        #     for loc in elem.dof_loc[tdim]:
        #         cols = mesh.cell[tdim]
        #         rows = np.tile(np.arange(num_elem), (1, tdim+1))
        #         vals = np.tile(loc[np.newaxis], (num_elem, 1))
        #         arr = csr_array((vals.reshape(-1), (rows.reshape(-1), cols.reshape(-1))), \
        #                         shape=(num_elem, mesh.point.shape[0]))
        #         all_locs = np.vstack((all_locs, arr @ mesh.point))
        #     # no need to match
        #     # broadcast to dof types and record
        #     num_new_dof = num_elem * num_dof_loc # with dof types excluded
        #     num_dof_type = len(elem.dof_name[tdim])
        #     self.dof_loc = np.vstack((self.dof_loc, np.repeat(all_locs, repeats=num_dof_type, axis=0)))
        #     for i, name in enumerate(elem.dof_name[tdim]):
        #         new_dof_idx = offset + np.arange(num_new_dof) * num_dof_type + i
        #         if name not in self.dof_group:
        #             self.dof_group[name] = new_dof_idx
        #         else:
        #             self.dof_group[name] = np.vstack((self.dof_group[name], new_dof_idx))
        #         new_elem_dof = np.zeros((num_dof_type, num_dof_loc, num_elem, 1))
        #         new_elem_dof
        self.num_dof = offset

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