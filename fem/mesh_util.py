import numpy as np
from .mesh import Mesh
from .element import LineP1, TriP1, LineP2, TriP2, VectorElement
from .funcspace import FunctionSpace
from .function import Function
from .refdom import RefTri
from tools.binsearchkw import binsearchkw

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

def splitRefine(mesh: Mesh) -> Mesh:
    fine_mesh = Mesh()
    fine_mesh.tdim = mesh.tdim
    fine_mesh.gdim = mesh.gdim
    N = (mesh.point.shape[0], *[c.shape[0] for c in mesh.cell[1:]])
    if mesh.tdim == 1:
        # set points
        fine_mesh.point = np.empty((N[0]+N[1], mesh.gdim))
        fine_mesh.point[:N[0]] = mesh.point
        fine_mesh.point[N[0]:] = 0.5 * (mesh.point[mesh.cell[1][:,0]] + mesh.point[mesh.cell[1][:,1]])
        fine_mesh.point_tag = np.zeros((N[0]+N[1], ), dtype=np.int32)
        fine_mesh.point_tag[:N[0]] = mesh.point_tag
        fine_mesh.cell[0] = mesh.cell[0].copy()
        fine_mesh.cell_tag[0] = mesh.cell_tag[0].copy()
        # set edges
        fine_mesh.cell[1] = np.empty((N[1] * 2, 2), dtype=np.int32)
        fine_mesh.cell[1][::2, 0] = mesh.cell[1][:,0]
        fine_mesh.cell[1][::2, 1] = N[0] + np.arange(N[1])
        fine_mesh.cell[1][1::2, 0] = N[0] + np.arange(N[1])
        fine_mesh.cell[1][1::2, 1] = mesh.cell[1][:,1]
        fine_mesh.cell_tag[1] = np.repeat(mesh.cell_tag[1], repeats=2)
    elif mesh.tdim == 2:
        # get all edges
        all_edges = RefTri._get_sub_entities(mesh.cell[2], dim=1) # (Nt, 3, 2)
        all_edges = all_edges.reshape(-1, 2) # (Nt*3, 2)
        all_edges.sort(axis=1)
        new_edges, inv_idx = np.unique(all_edges, return_inverse=True, axis=0)
        # uq_edges: (Ne, 2), inv_idx: (Nt*3, )
        inv_idx = N[0] + inv_idx.reshape(N[2], 3)
        Ne = new_edges.shape[0]
        # set points
        fine_mesh.point = np.empty((N[0] + Ne, mesh.gdim))
        fine_mesh.point[:N[0]] = mesh.point
        fine_mesh.point[N[0]:] = 0.5 * (mesh.point[new_edges[:,0]] + mesh.point[new_edges[:,1]])
        fine_mesh.point_tag = np.zeros((N[0] + Ne, ), dtype=np.int32)
        fine_mesh.point_tag[:N[0]] = mesh.point_tag
        fine_mesh.cell[0] = mesh.cell[0].copy()
        fine_mesh.cell_tag[0] = mesh.cell_tag[0].copy()
        # set edges
        sorted_edge = np.sort(mesh.cell[1], axis=1)
        e_idx = binsearchkw(new_edges, sorted_edge)
        assert np.all(e_idx != -1)
        fine_mesh.cell[1] = np.empty((N[1] * 2, 2), dtype=np.int32)
        fine_mesh.cell[1][::2, 0] = mesh.cell[1][:, 0]
        fine_mesh.cell[1][::2, 1] = N[0] + e_idx
        fine_mesh.cell[1][1::2, 0] = N[0] + e_idx
        fine_mesh.cell[1][1::2, 1] = mesh.cell[1][:, 1]
        fine_mesh.cell_tag[1] = np.repeat(mesh.cell_tag[1], repeats=2)
        # set triangles
        fine_mesh.cell[2] = np.empty((N[2]*4, 3), dtype=np.int32)
        fine_mesh.cell[2][::4, 0] = mesh.cell[2][:,0]
        fine_mesh.cell[2][::4, 1] = inv_idx[:,0]
        fine_mesh.cell[2][::4, 2] = inv_idx[:,2]
        fine_mesh.cell[2][1::4, 0] = mesh.cell[2][:,1]
        fine_mesh.cell[2][1::4, 1] = inv_idx[:,1]
        fine_mesh.cell[2][1::4, 2] = inv_idx[:,0]
        fine_mesh.cell[2][2::4, 0] = mesh.cell[2][:,2]
        fine_mesh.cell[2][2::4, 1] = inv_idx[:,2]
        fine_mesh.cell[2][2::4, 2] = inv_idx[:,1]
        fine_mesh.cell[2][3::4, :] = inv_idx
        fine_mesh.cell_tag[2] = np.repeat(mesh.cell_tag[2], repeats=4)
    else:
        raise RuntimeError(f"Cannot refine a mesh of dimension {mesh.tdim}.")
    fine_mesh.build_facet_ref()
    return fine_mesh
