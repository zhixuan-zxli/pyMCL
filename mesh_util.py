from typing import Optional
import numpy as np
from mesh import Mesh
from fe import TriP1, LineP2, TriP2
from function import Function

def splitRefine(mesh: Mesh) -> Mesh:
    assert mesh.tdim == 1 or mesh.tdim == 2
    P2 = TriP2(mesh) if mesh.tdim == 2 else LineP2(mesh)
    Np = mesh.point.shape[0]
    Ne = mesh.cell[1].shape[0]
    Nt = mesh.cell[2].shape[0] if mesh.tdim == 2 else 0
    fine_mesh = Mesh()
    fine_mesh.tdim = mesh.tdim
    fine_mesh.gdim = mesh.gdim
    # set up the point
    fine_mesh.point = P2.dofloc.copy()
    fine_mesh.point_tag = np.zeros((P2.num_dof, ), dtype=np.uint32)
    fine_mesh.point_tag[:Np] = mesh.point_tag
    # set up the edges
    fine_mesh.cell[1] = np.zeros((2, Ne, 3), dtype=np.uint32)
    fine_mesh.cell[1][0, :, :] = P2.cell_dof[1][:, [0,2,3]]
    fine_mesh.cell[1][1, :, :] = P2.cell_dof[1][:, [2,1,3]]
    fine_mesh.cell[1] = fine_mesh.cell[1].reshape((2*Ne, 3))
    # set up the triangles
    if mesh.tdim >= 2:
        fine_mesh.cell[2] = np.zeros((4, Nt, 4), dtype=np.uint32)
        fine_mesh.cell[2][0, :, :] = P2.cell_dof[2][:, [0,3,5,6]]
        fine_mesh.cell[2][1, :, :] = P2.cell_dof[2][:, [3,1,4,6]]
        fine_mesh.cell[2][2, :, :] = P2.cell_dof[2][:, [5,4,2,6]]
        fine_mesh.cell[2][3, :, :] = P2.cell_dof[2][:, [3,4,5,6]]
        fine_mesh.cell[2] = fine_mesh.cell[2].reshape((4*Nt, 4))
    return fine_mesh

def setMeshMapping(mesh: Mesh, mapping: Optional[Function] = None):
    if mapping is None:
        # set an affine mapping
        if mesh.tdim == 2:
            mesh.coord_fe = TriP1(mesh, mesh.gdim)
            mesh.coord_map = Function(mesh.coord_fe)
            np.copyto(mesh.coord_map, mesh.point)
        else:
            raise NotImplementedError
    else:
        mesh.mapping = mapping
