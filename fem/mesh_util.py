import numpy as np
from .mesh import Mesh
from .element import LineP1, TriP1, LineP2, TriP2, VectorElement
from .funcspace import FunctionSpace
from .function import Function

# def splitRefine(mesh: Mesh) -> Mesh:
#     assert mesh.tdim == 1 or mesh.tdim == 2
#     P2 = TriP2(mesh) if mesh.tdim == 2 else LineP2(mesh)
#     Np = mesh.point.shape[0]
#     Ne = mesh.cell[1].shape[0]
#     Nt = mesh.cell[2].shape[0] if mesh.tdim == 2 else 0
#     fine_mesh = Mesh()
#     fine_mesh.tdim = mesh.tdim
#     fine_mesh.gdim = mesh.gdim
#     # set up the point
#     fine_mesh.point = P2.dofloc.copy()
#     fine_mesh.point_tag = np.zeros((P2.num_dof, ), dtype=np.uint32)
#     fine_mesh.point_tag[:Np] = mesh.point_tag
#     # set up the edges
#     fine_mesh.cell[1] = np.zeros((2, Ne, 3), dtype=np.uint32)
#     fine_mesh.cell[1][0, :, :] = P2.cell_dof[1][:, [0,2,3]]
#     fine_mesh.cell[1][1, :, :] = P2.cell_dof[1][:, [2,1,3]]
#     fine_mesh.cell[1] = fine_mesh.cell[1].reshape((2*Ne, 3))
#     # set up the triangles
#     if mesh.tdim >= 2:
#         fine_mesh.cell[2] = np.zeros((4, Nt, 4), dtype=np.uint32)
#         fine_mesh.cell[2][0, :, :] = P2.cell_dof[2][:, [0,3,5,6]]
#         fine_mesh.cell[2][1, :, :] = P2.cell_dof[2][:, [3,1,4,6]]
#         fine_mesh.cell[2][2, :, :] = P2.cell_dof[2][:, [5,4,2,6]]
#         fine_mesh.cell[2][3, :, :] = P2.cell_dof[2][:, [3,4,5,6]]
#         fine_mesh.cell[2] = fine_mesh.cell[2].reshape((4*Nt, 4))
#     return fine_mesh

def setMeshMapping(mesh: Mesh, order: int = 1):
    """
    Set up the P-x (x=1,2) isoparametric mapping for the mesh. 
    """
    elem_tab = {(1,1): LineP1, (1,2): LineP2, (2,1): TriP1, (2,2): TriP2}
    # set an affine mapping
    try:
        elem = elem_tab[(mesh.tdim, order)]
    except KeyError:
        raise RuntimeError(f"Cannot construct an isoparametric element of order {order}. ")
    mesh.coord_fe = FunctionSpace(mesh, VectorElement(elem, mesh.gdim))
    mesh.coord_map = Function(mesh.coord_fe)
    for d in range(mesh.gdim):
        dof_d = mesh.coord_fe.dof_group["u_" + str(d)]
        mesh.coord_map[dof_d] = mesh.coord_fe.dof_loc[dof_d, d]
