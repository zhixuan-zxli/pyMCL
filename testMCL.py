import numpy as np
from math import cos
from fem import *
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot


class PhysicalParameters:
    eta_2: float = 0.1
    mu_1: float = 0.1
    mu_2: float = 0.1
    mu_cl: float = 0.1
    cosY: float = cos(np.pi*2.0/3)
    B: float = 1e-1
    Y: float = 1e1

class SolverParemeters:
    dt: float = 1.0/1024
    Te: float = 1.0/8
    startStep: int = 0
    stride: int = 1
    numChekpoint: int = 0
    vis: bool = True

# ===========================================================
# bilinear forms for the elastic sheet

@BilinearForm
def c_L2(w: QuadData, q: QuadData, x: QuadData) -> np.ndarray:
    return np.sum(w * q, axis=0, keepdims=True) * x.dx

@LinearForm
def l_dq(w: QuadData, x: QuadData, q_k: QuadData) -> np.ndarray:
    return np.sum(w * q_k.grad[:,0], axis=0, keepdims=True) * x.dx

@BilinearForm
def c_phitau(phi: QuadData, tau: QuadData, x: QuadData) -> np.ndarray:
    # phi, tau: (2, Nf, Nq)
    return np.sum(phi * tau, axis=0, keepdims=True) * x.dx

@BilinearForm
def c_phim3(phi: QuadData, m3: QuadData, x: QuadData) -> np.ndarray:
    return np.sum(phi * m3, axis=0, keepdims=True) * x.ds

@BilinearForm
def c_cl_phim(phi: QuadData, m: QuadData, xphi: QuadData, _: QuadData) -> np.ndarray:
    # phi: (2, Nf, Nq)
    # m.grad: (1, 2, Nf, Nq)
    # xphi.fn: (2, Nf, Nq)
    return (m[0] * phi.grad[1,0] * xphi.fn[0])[np.newaxis] * xphi.ds
    # return (phi[1] * m.grad[0,0] * xm.fn[0])[np.newaxis] * xm.ds

@BilinearForm
def c_phim(phi: QuadData, m: QuadData, x: QuadData) -> np.ndarray:
    # phi.grad: (2, 2, Nf, Nq)
    # m.grad: (1, 2, Nf, Nq)
    return (phi.grad[1,0] * m.grad[0,0])[np.newaxis] * x.dx

@BilinearForm
def c_phiq(phi: QuadData, w: QuadData, x: QuadData, w_k: QuadData) -> np.ndarray:
    # Here the trial function w is the displacement, 
    # and the argument w_k, q_k is the push-forward displacement, deformation (resp.) 
    # from the last time step. 
    c1 = (w.grad[0,0] + 0.5 * w_k.grad[1,0] * w.grad[1,0]) * phi.grad[0,0]
    c2 = (w_k.grad[0,0] + 0.5 * w_k.grad[1,0]**2) * w.grad[1,0] * phi.grad[1,0]
    return (c1+c2)[np.newaxis] * x.dx

@BilinearForm
def c_phiq_mf(phi: QuadData, w: QuadData, x: QuadData, gamma: np.ndarray) -> np.ndarray:
    # Again w is the displacement; 
    # x is the surface measure on the deformed surface from the last time step; 
    # id is the identify mapping for the current reference domain, so that w + id is the current deformation; 
    # gamma is the piecewise constant surface tension. 
    # phi.grad, w.grad, id.grad: (2, 2, Nf, Nq)
    return np.sum(phi.grad * w.grad, axis=(0,1))[np.newaxis] \
        * gamma[np.newaxis, np.newaxis, :, np.newaxis] * x.dx

# For c_wm, use c_L2
# For c_wq, use the transpose of c_phim
# for c_cl_wq, use the transpose of c_cl_phim

# ===========================================================
# Forms for the dynamics of the sheet

# for b_piq, use c_L2. 
# for b_piu, use a_xitau. 

@LinearForm
def l_pi(pi: QuadData, x: QuadData, q_k: QuadData, eta_k: QuadData) -> np.ndarray:
    # q_k: the deformation of the last time step, 
    # eta_k: the mesh velocity
    # x: the surface measure over the reference sheet in the last time step
    return np.sum(q_k.grad[:,0] * eta_k * pi, axis=0, keepdims=True) * x.dx

@BilinearForm
def b_pitau(pi: QuadData, tau: QuadData, x: QuadData, q_k: QuadData, mu_i: np.ndarray) -> np.ndarray:
    # q_k: the deformation of the last time step
    # q_k.grad: (2, 1, Ne, Nq)
    # tau: (2, Ne, Nq)
    # mu_i: the slip coefficient
    # x: the surface measure over the reference sheet in the last time step
    m = q_k.grad[:,0]
    m = m / np.linalg.norm(m, ord=None, axis=0, keepdims=True) # (2, Ne, Nq)
    Ptau =  np.sum(m * tau, axis=0, keepdims=True) # (1, Ne, Nq)
    Ppi = np.sum(m * pi, axis=0, keepdims=True) # (1, Ne, Nq)
    return Ptau * Ppi * (1.0/mu_i[np.newaxis, :, np.newaxis]) * x.dx

# ===========================================================
# bilinear forms for the fluid and fluid interface

@BilinearForm
def a_xiu(xi: QuadData, u: QuadData, x: QuadData, eta) -> np.ndarray:
    # eta: (Ne,)
    # grad: (2, 2, Ne, Nq)    
    z = np.zeros(x.shape[1:]) # (Ne, Nq)
    for i, j in (0,0), (0,1), (1,0), (1,1):
        z += (u.grad[i,j] + u.grad[j,i]) * xi.grad[i,j]
    return (z * eta[:, np.newaxis])[np.newaxis] * x.dx

@BilinearForm
def a_xip(xi: QuadData, p: QuadData, x: QuadData) -> np.ndarray:
    # w.grad: (2,2,Ne,Nq)
    # p: (1, 1, Nq)
    z = (xi.grad[0,0] + xi.grad[1,1]) * p[0]
    return z[np.newaxis] * x.dx

@BilinearForm
def a_xik(xi: QuadData, kappa: QuadData, x: QuadData) -> np.ndarray:
    # kappa: (1, 1, Nq)
    # w: (2, 1, Nq)
    # x.fn: (2, Nf, Nq)
    return np.sum(x.fn * xi, axis=0, keepdims=True) * kappa * x.ds # (2, Nf, Nq)

@BilinearForm
def a_xitau(xi: QuadData, tau: QuadData, x: QuadData) -> np.ndarray:
    # xi: (2, Nf, Nq)
    # tau: (2, Nf, Nq)
    return np.sum(xi * tau, axis=0, keepdims=True) * x.ds

@BilinearForm
def a_zy(z: QuadData, y: QuadData, x: QuadData) -> np.ndarray:
    return np.sum(z.grad * y.grad, axis=(0,1))[np.newaxis] * x.dx

@BilinearForm
def a_zk(z: QuadData, k: QuadData, x: QuadData) -> np.ndarray:
    # k: (1, Nf, Nq)
    return np.sum(z * x.cn, axis=0, keepdims=True) * k * x.dx 

@BilinearForm
def a_zm3(z: QuadData, m3: QuadData, x: QuadData) -> np.ndarray:
    # z, m3: (2, Nf, Nq=1)
    return np.sum(z * m3, axis=0, keepdims=True) * x.ds

# For the constraint of attaching contact line, 
# (i.e. the equation for m3), 
# the bilinear form should be the same as a_zm3. 


# ===========================================================


if __name__ == "__main__":

    phyp = PhysicalParameters()
    solp = SolverParemeters()

    # physical groups from GMSH
    # group_name = {"fluid_1": 1, "fluid_2": 2, "interface": 3, "dry": 4, "wet": 5, \
    #              "right": 6, "top": 7, "left": 8, "cl": 9, "clamp": 10}
    mesh = Mesh()
    mesh.load("mesh/two-phase.msh")
    setMeshMapping(mesh)
    i_mesh = mesh.view(1, tags=(3, )) # interface mesh
    setMeshMapping(i_mesh)
    s_mesh = mesh.view(1, tags=(4, 5)) # sheet reference mesh
    setMeshMapping(s_mesh)
    cl_mesh = mesh.view(0, tags=(9, )) # contact line mesh
    setMeshMapping(cl_mesh)
    # for enforcing periodic constraint
    def periodic_constraint(x: np.ndarray) -> np.ndarray:
        flag = np.abs(x[:,0] - 1.0) < 1e-12
        x[flag,0] -= 2.0

    sheet_P2v = FunctionSpace(s_mesh, VectorElement(LineP2, 2))
    sheet_P1v = s_mesh.coord_fe # type: FunctionSpace # should be FunctionSpace(s_mesh, VectorElement(LineP1, 2))
    cl_vsp = FunctionSpace(cl_mesh, VectorElement(NodeElement, 2))
    assert cl_vsp.dof_loc[0,0] < cl_vsp.dof_loc[2,0]

    q = Function(sheet_P2v)
    q_k = Function(sheet_P2v)
    q_k[::2] = sheet_P2v.dof_loc[::2, 0]
    q_k[1::2] = (q_k[::2] + 1.0) * (q_k[::2] - 1.0)
    
    # extract the CL dofs
    cl_dof_P2v = np.unique(sheet_P2v.getFacetDof((9, )))
    cl_dof_P1v = np.unique(sheet_P1v.getFacetDof((9, )))

    chi_k = s_mesh.coord_map[cl_dof_P1v[::2]].view(np.ndarray)
    chi = np.zeros_like(chi_k)
    m3_k = Function(cl_vsp)
    m3_k[1::2] = -1.0 # manual set
    
    # set up the measures and the function basis
    dA = Measure(s_mesh, 1, order=5)
    sheet_P1v_basis = FunctionBasis(sheet_P1v, dA)

    # =================================================================
    # Step 1. Update the reference contact line. 
    # project the discontinuous deformation gradient onto P1 to find the conormal vector m1

    C_L2 = c_L2.assemble(sheet_P1v_basis, sheet_P1v_basis, dA)
    L_DQ = l_dq.assemble(sheet_P1v_basis, dA, q_k = q_k._interpolate(dA))

    dq_k = Function(sheet_P1v)
    dq_k[:] = spsolve(C_L2, L_DQ)

    # extract the conormal at the contact line
    dq_k_at_cl = dq_k.view(np.ndarray)[cl_dof_P1v].reshape(-1, 2) # (-1, 2)
    m1_k = dq_k_at_cl / np.linalg.norm(dq_k_at_cl, ord=None, axis=1, keepdims=True) # (2, 2)
    # find the correct direction of m1
    a = sheet_P1v.dof_loc[cl_dof_P1v[0],0] > sheet_P1v.dof_loc[cl_dof_P1v[2],0] # (1, )
    m1_k[int(a)] = -m1_k[int(a)] 
    # find the displacement of the reference CL
    a = np.sum(dq_k_at_cl * m1_k, axis=1) #(2, )
    m3_k_ = m3_k.view(np.ndarray).reshape(2,2)
    cl_disp = - solp.dt / (phyp.mu_cl * a) * (phyp.cosY + 1.0 * np.sum(m3_k_ * m1_k, axis=1)) 
    chi = chi_k + cl_disp
    
    # =================================================================
    # Step 2. Find the sheet mesh displacement. 
    id_k = Function(s_mesh.coord_fe)
    id_k[:] = s_mesh.coord_map # save the previous reference sheet mesh

    xx = s_mesh.coord_map.view(np.ndarray)[::2]
    s_mesh_disp = np.where(
        xx <= chi_k[0], (xx + 1.0) / (chi_k[0] + 1.0) * cl_disp[0], 
        np.where(xx >= chi_k[1], (1.0 - xx) / (1.0 - chi_k[1]) * cl_disp[1], 
                 (cl_disp[0] * (chi_k[1] - xx) + cl_disp[1] * (xx - chi_k[0])) / (chi_k[1] - chi_k[0]))
    )
    s_mesh_vel = s_mesh_disp / solp.dt
    s_mesh.coord_map[::2] += s_mesh_disp

    # =================================================================
    # Step 3. Solve the fluid, the fluid-fluid interface, and the sheet deformation. 

    # prepare the variable coefficients
    viscosity = np.where(mesh.cell_tag[2] == 1, 1.0, phyp.eta_2)
    slip_fric = np.where(s_mesh.cell_tag[1] == 5, phyp.mu_1, phyp.mu_2)

    # set up the function spaces
    U_sp = FunctionSpace(mesh, VectorElement(TriP2, 2), constraint=periodic_constraint)
    P1_sp = FunctionSpace(mesh, TriP1, constraint=periodic_constraint)
    P0_sp = FunctionSpace(mesh, TriDG0)
    Y_sp = i_mesh.coord_fe # type: FunctionSpace # should be FunctionSpace(i_mesh, VectorElement(LineP1, 2))
    K_sp = FunctionSpace(i_mesh, LineP1)
    # cl_vsp is the function space for m3
    Q_sp = sheet_P2v # VectorElement(LineP2, 2) # for deformation and also for the fluid stress
    M_sp = FunctionSpace(s_mesh, LineP2)
    M3_sp = cl_vsp # VectorElement(NodeElement, 2)
    
    # set up the measures
    dx = Measure(mesh, dim=2, order=3)
    ds_i = Measure(mesh, dim=1, order=3, tags=(3, )) # the fluid interface restricted from the bulk mesh
    ds = Measure(i_mesh, dim=1, order=3)
    da_x = Measure(mesh, dim=1, order=5, tags=(4, 5)) # the deformed sheet restricted from the bulk mesh
    dp_i = Measure(i_mesh, dim=0, order=1, tags=(9, )) # the CL restricted from the fluid interfacef
    dp_s = Measure(s_mesh, dim=0, order=1, tags=(9, )) # the CL on the reference sheet mesh
    dp = Measure(cl_mesh, dim=0, order=1)

    dA_k = Measure(s_mesh, dim=1, order=5, coord_map=id_k) # the reference sheet mesh at the last time step
    dA = Measure(s_mesh, dim=1, order=5) # the reference sheet surface measure
    da = Measure(s_mesh, dim=1, order=5, coord_map=q_k) # the deformed sheet surface measure

    # set up the function bases
    u_basis = FunctionBasis(U_sp, dx)
    p1_basis = FunctionBasis(P1_sp, dx)
    p0_basis = FunctionBasis(P0_sp, dx)
    u_i_basis = FunctionBasis(U_sp, ds_i)
    u_b_basis = FunctionBasis(U_sp, da_x)

    tau_basis = FunctionBasis(Q_sp, da)

    y_basis = FunctionBasis(Y_sp, ds)
    y_cl_basis = FunctionBasis(Y_sp, dp_i)
    k_basis = FunctionBasis(K_sp, ds)

    m3_basis = FunctionBasis(M3_sp, dp)

    q_k_basis = FunctionBasis(Q_sp, dA_k) # also the basis for tau
    q_cl_basis = FunctionBasis(Q_sp, dp_s)

    # assembly by blocks
    A_XIU = a_xiu.assemble(u_basis, u_basis, dx, eta=viscosity)
    A_XIP1 = a_xip.assemble(u_basis, p1_basis, dx)
    A_XIP0 = a_xip.assemble(u_basis, p0_basis, dx)
    A_XIK = a_xik.assemble(u_i_basis, k_basis, ds_i)
    A_XITAU = a_xitau.assemble(u_b_basis, tau_basis, da_x)

    A_ZY = a_zy.assemble(y_basis, y_basis, ds)
    A_ZK = a_zk.assemble(y_basis, k_basis, ds)
    A_ZM3 = a_zm3.assemble(y_cl_basis, m3_basis, dp_i)
    
    A_M3Q = a_zm3.assemble(m3_basis, q_cl_basis, dp_s)

    B_PIQ = c_L2.assemble(q_k_basis, q_k_basis, dA_k)
    B_PIU = a_xitau.assemble(q_k_basis, u_b_basis, dA_k) # create a form with x.dx
    # Note: although u_b_basis is not on dA_k, 
    # we do not need the gradient. 
    B_PITAU = b_pitau.assemble(q_k_basis, q_k_basis, dA_k, mu_i=slip_fric)

    pass
    
