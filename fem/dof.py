    
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