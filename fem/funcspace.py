import numpy as np
from .mesh import Mesh
from .element import Element

class FunctionSpace:

    mesh: Mesh
    elem: Element
    periodic: bool
    
    entity_dof: list[np.ndarray]
    elem_dof: np.ndarray
    num_dof: int

    # dofloc

    def __init__(self, mesh: Mesh, elem: Element, periodic: bool = False) -> None:
        self.mesh = mesh
        self.elem = elem
        self.periodic = periodic

        tdim = elem.tdim
        assert mesh.tdim == tdim

        # build the dof for entities of each dimension
        offset = 0
        self.entity_dof = [None] * (tdim+1)
        for d in range(tdim+1):
            num_entity = mesh.cell[d].shape[0] if d > 0 else mesh.point.shape[0]
            per_ent = elem.num_dof_per_ent[d]
            self.entity_dof[d] = \
                np.arange(per_ent * num_entity, dtype=np.int32) \
                .reshape(per_ent, num_entity) + offset
            offset += num_entity * per_ent
        self.num_dof = offset

        # collect the dof for each element
        num_elem = mesh.cell[tdim].shape[0]
        self.elem_dof = np.zeros((0, num_elem), dtype=np.int32)
        for d in range(tdim):
            num_sub_ent = mesh.cell_entity[d].shape[1]
            for i in range(num_sub_ent):
                self.elem_dof = np.vstack(
                    (self.elem_dof, 
                    self.entity_dof[d][:, mesh.cell_entity[d][:,i]])
                )
        self.elem_dof = np.vstack((self.elem_dof, self.entity_dof[-1]))


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