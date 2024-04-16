import numpy as np
from math import cos
from fem import *
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot


class PhysicalParameters:
    eta_2: float = 0.1
    mu_1: float = 1.0
    mu_2: float = 1.0
    mu_cl: float = 1.0
    cosY: float = cos(np.pi*2.0/3)
    B: float = 1e-2

class SolverParemeters:
    dt: float = 1.0/1024
    Te: float = 1.0/8
    startStep: int = 0
    stride: int = 1
    numChekpoint: int = 0
    vis: bool = True

@BilinearForm
def c_lap(w: QuadData, q: QuadData, x: QuadData) -> np.ndarray:
    return w.grad[0] * q.grad[0] * x.dx

@BilinearForm
def c_H0(w: QuadData, q: QuadData, x: QuadData) -> np.ndarray:
    return w * q * x.dx

@BilinearForm
def c_cl(w: QuadData, q: QuadData, _, xq: QuadData) -> np.ndarray:
    return w * (q.grad[0] * xq.fn) * xq.ds

@LinearForm
def l_ver(phi: QuadData, x: QuadData) -> np.ndarray:
    return 0.5 * phi * x.ds


if __name__ == "__main__":

    phyp = PhysicalParameters()
    solp = SolverParemeters()

    # physical groups from GMSH
    # group_name = {"fluid_1": 1, "fluid_2": 2, "interface": 3, "dry": 4, "wet": 5, \
    #              "right": 6, "top": 7, "left": 8, "cl": 9, "clamp": 10}
    mesh = Mesh()
    mesh.load("mesh/two-phase.msh")
    # setMeshMapping(mesh, 2)
    # i_mesh = mesh.view(1, tags=(3, )) # interface mesh
    # setMeshMapping(i_mesh)
    s_mesh = mesh.view(1, tags=(4,5)) # sheet reference mesh
    setMeshMapping(s_mesh)
    # cl_mesh = mesh.view(0, sub_ids=(9, )) # contact line mesh
    # setMeshMapping(cl_mesh)

    sheet_def_space = FunctionSpace(s_mesh, VectorElement(LineP2, 2))
    sheet_grad_space = s_mesh.coord_fe # type: FunctionSpace # should be FunctionSpace(s_mesh, VectorElement(LineP1, 2))
    q_k = Function(sheet_def_space)
    q_k[::2] = sheet_def_space.dof_loc[::2, 0]
    q_k[1::2] = (q_k[::2] + 1.0) * (q_k[::2] - 1.0)
    
    # extract the CL dofs
    cl_dof_in_def = np.unique(sheet_def_space.getFacetDof((9, )))
    cl_dof_in_grad = np.unique(sheet_grad_space.getFacetDof((9, )))

    chi_k = np.zeros(2)
    chi_k[:] = s_mesh.coord_map[cl_dof_in_grad[::2]]
    chi = np.zeros(2)
    m3_k = np.zeros((2, 2))
    m3_k[:,1] = -1.0
    
    # set up the measures and the function basis
    ds = Measure(s_mesh, 1, order=5)
    sheet_grad_basis = FunctionBasis(sheet_grad_space, ds)

    # =================================================================
    # Step 1. Update the reference contact line. 
    # project the discontinuous deformation gradient onto P1 to find the conormal vector m1
    @BilinearForm
    def c_H0(w: QuadData, q: QuadData, x: QuadData) -> np.ndarray:
        return np.sum(w * q, axis=0, keepdims=True) * x.dx
    @LinearForm
    def l_dq(w: QuadData, x: QuadData, q_k: QuadData) -> np.ndarray:
        return np.sum(w * q_k.grad[:,0], axis=0, keepdims=True) * x.dx

    C_H0 = c_H0.assemble(sheet_grad_basis, sheet_grad_basis, ds)
    L_DQ = l_dq.assemble(sheet_grad_basis, ds, q_k = q_k._interpolate(ds))

    dq_k = Function(sheet_grad_space)
    dq_k[:] = spsolve(C_H0, L_DQ)

    # extract the conormal at the contact line
    dq_k_at_cl = dq_k.view(np.ndarray)[cl_dof_in_grad].reshape(-1, 2) # (-1, 2)
    m1_k = dq_k_at_cl / np.linalg.norm(dq_k_at_cl, ord=None, axis=1, keepdims=True) # (2, 2)
    # find the correct direction of m1
    a = sheet_grad_space.dof_loc[cl_dof_in_grad[0],0] > sheet_grad_space.dof_loc[cl_dof_in_grad[2],0] # (1, )
    m1_k[int(a)] = -m1_k[int(a)] 
    # find the displacement of the reference CL
    a = np.sum(dq_k_at_cl * m1_k, axis=1) #(2, )
    cl_disp = - solp.dt / (phyp.mu_cl * a) * (phyp.cosY + 1.0 * np.sum(m3_k * m1_k, axis=1)) 
    chi = chi_k + cl_disp
    
    # =================================================================
    # Step 2. Find the sheet mesh displacement. 
    xx = s_mesh.coord_map[::2]
    s_mesh_disp = np.where(
        xx <= chi_k[0], (xx + 1.0) / (chi_k[0] + 1.0) * cl_disp[0], 
        np.where(xx >= chi_k[1], (1.0 - xx) / (1.0 - chi_k[1]) * cl_disp[1], 
                 (cl_disp[0] * (chi_k[1] - xx) + cl_disp[1] * (xx - chi_k[0])) / (chi_k[1] - chi_k[0]))
    )
    s_mesh_vel = s_mesh_disp / solp.dt
    s_mesh.coord_map[::2] += s_mesh_disp

    # q_k_ = q_k.view(np.ndarray)
    # pyplot.plot(q_k[::2], q_k[1::2], 'ro')
    # pyplot.quiver(q_k_[cl_dof_from_def[::2]], q_k_[cl_dof_from_def[1::2]], temp[:,0], temp[:,1])
    # pyplot.show()

    
