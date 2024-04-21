from os import mkdir
import argparse
import numpy as np
from math import cos, ceil
from fem import *
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot
from colorama import Fore, Style


class PhysicalParameters:
    eta_2: float = 0.1
    mu_1: float = 0.1
    mu_2: float = 0.1
    mu_cl: float = 1.0
    cosY: float = cos(np.pi*2.0/3)
    gamma_1: float = 0.0
    gamma_3: float = 10.0
    gamma_2: float = 0.0 + 10.0 * cos(np.pi*2.0/3) # to be consistent: gamma_2 = gamma_1 + gamma_3 * cos(theta_Y)
    B: float = 1.0
    Y: float = 1e3 #1e1

class SolverParemeters:
    dt: float = 1.0/1024
    Te: float = 256.0/1024 #1.0/8
    resume: bool = False
    stride: int = 64
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
def c_cl_phim(phi: QuadData, m: QuadData, xphi: QuadData, _: QuadData) -> np.ndarray:
    # phi: (2, Nf, Nq)
    # m.grad: (1, 2, Nf, Nq)
    # xphi.fn: (2, Nf, Nq)
    return 0.5 * (m[0] * phi.grad[1,0] * xphi.fn[0])[np.newaxis] * xphi.ds
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
def c_phiq_surf(phi: QuadData, w: QuadData, x: QuadData, gamma: np.ndarray) -> np.ndarray:
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
# for b_piu, use c_L2. 

@LinearForm
def l_pi_adv(pi: QuadData, x: QuadData, q_k: QuadData, eta_k: QuadData) -> np.ndarray:
    # q_k: the deformation of the last time step, 
    # eta_k: the mesh velocity, (2, Nf, Nq)
    # x: the surface measure over the reference sheet in the last time step
    return np.sum(q_k.grad[:,0] * eta_k[0][np.newaxis] * pi, axis=0, keepdims=True) * x.dx

@LinearForm
def l_pi_L2(pi: QuadData, x: QuadData, q_k: QuadData) -> np.ndarray:
    return np.sum(pi * q_k, axis=0, keepdims=True) * x.dx

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

@LinearForm
def l_m3(m3: QuadData, x: QuadData, id: QuadData) -> np.ndarray:
    # m3: (2, Nf, Nq)
    # id: (2, Nf, Nq)
    return np.sum(m3 * id, axis=0, keepdims=True) * x.ds

# ===========================================================
# linear elasticity for the bulk mesh displacement
@BilinearForm
def a_el(Z: QuadData, Y: QuadData, x: QuadData) -> np.ndarray:
    # grad: (2, 2, Ne, Nq)
    # x.dx: (1, Ne, Nq)
    lam_dx = x.dx + (x.dx.max() - x.dx.min()) # (1, Ne, Nq)
    r = np.zeros(x.shape[1:]) # (Ne,Nq)
    tr = Y.grad[0,0] + Y.grad[1,1] # (Ne, Nq)
    for i, j in (0,0), (0,1), (1,0), (1,1):
        r += (Y.grad[i,j] + Y.grad[j,i] + (i==j) * tr) * Z.grad[i,j]
    return r[np.newaxis] * lam_dx

# ===========================================================

def lift_to_P2(P2_space: FunctionSpace, p1_func: Function) -> Function:
    p2_func = Function(P2_space)
    p2_func[:p1_func.size] = p1_func
    rdim = P2_space.elem.rdim
    assert rdim == p1_func.fe.elem.rdim
    if P2_space.mesh.tdim == 1:
        for d in range(rdim):
            p2_func[P2_space.elem_dof[2*rdim+d]] = 0.5 * (p1_func[P2_space.elem_dof[d]] + p1_func[P2_space.elem_dof[rdim+d]])
    else:
        raise NotImplementedError
    return p2_func

def down_to_P1(P1_space: FunctionSpace, p2_func: Function) -> Function:
    p1_func = Function(P1_space)
    p1_func[:] = p2_func[:P1_space.num_dof]
    return p1_func


# ===========================================================


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", help="Resume from previous computations.", action="store_true")
    args = parser.parse_args()
    phyp = PhysicalParameters()
    solp = SolverParemeters()
    solp.resume = args.resume

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

    # prepare the variable coefficients
    viscosity = np.where(mesh.cell_tag[2] == 1, 1.0, phyp.eta_2)
    slip_fric = np.where(s_mesh.cell_tag[1] == 5, phyp.mu_1, phyp.mu_2)
    surf_tens = np.where(s_mesh.cell_tag[1] == 5, phyp.gamma_1, phyp.gamma_2)

    # =================================================================
    # set up the function spaces
    U_sp = FunctionSpace(mesh, VectorElement(TriP2, 2), constraint=periodic_constraint)
    P1_sp = FunctionSpace(mesh, TriP1, constraint=periodic_constraint)
    P0_sp = FunctionSpace(mesh, TriDG0, constraint=periodic_constraint)
    Y_sp = i_mesh.coord_fe # type: FunctionSpace # should be VectorElement(LineP1, 2)
    K_sp = FunctionSpace(i_mesh, LineP1)
    Q_sp = FunctionSpace(s_mesh, VectorElement(LineP2, 2)) # for deformation and also for the fluid stress
    Q_P1_sp = s_mesh.coord_fe # type: FunctionSpace # VectorElement(LineP1, 2), for projection to continuous gradient
    MOM_sp = FunctionSpace(s_mesh, LineP2)
    M3_sp = FunctionSpace(cl_mesh, VectorElement(NodeElement, 2))

    # extract the useful DOFs
    cl_dof_Q2 = np.unique(Q_sp.getFacetDof(tags=(9,)))
    cl_dof_Q1 = np.unique(Q_P1_sp.getFacetDof(tags=(9,)))

    # declare the solution functions
    u = Function(U_sp)
    p1 = Function(P1_sp)
    p0 = Function(P0_sp)
    y = Function(Y_sp)
    y_k = Function(Y_sp)
    kappa = Function(K_sp)
    w = Function(Q_sp)   # the displacement
    w_k = Function(Q_sp) # the displacement
    tau = Function(Q_sp)
    mom = Function(MOM_sp)
    m3 = Function(M3_sp)
    m3[1::2] = -1.0 # need an initial value for m3
    
    id_k = Function(s_mesh.coord_fe)
    id = Function(s_mesh.coord_fe)

    # resume from checkpoints
    try:
        mkdir("MCL-output")
    except FileExistsError:
        pass
    if solp.resume:
        pass

    # create the figures
    if solp.vis:
        pyplot.ion()
        ax = pyplot.subplot()

    step = 0
    if solp.stride == 0 and solp.numChekpoint > 0:
        solp.stride = ceil(solp.Te / solp.dt / solp.numChekpoint)

    while True:
        # retrieve the mesh mapping; 
        # they will be needed in the measures. 
        y_k[:] = i_mesh.coord_map
        id_k[:] = s_mesh.coord_map 
        w_k[:] = w
        q_k = w_k + lift_to_P2(Q_sp, id_k) # type: Function

        t = step * solp.dt
        if solp.vis:
            ax.clear()
            ax.triplot(mesh.coord_map[::2], mesh.coord_map[1::2], triangles=mesh.coord_fe.elem_dof[::2,:].T//2)
            m3_ = m3.view(np.ndarray)
            ax.quiver(q_k[cl_dof_Q2[::2]], q_k[cl_dof_Q2[1::2]], m3_[::2], m3_[1::2])
            ax.plot(id_k[::2], -0.1*np.ones(id_k.size//2), 'b+') # plot reference sheet mesh
            ax.plot(id_k[cl_dof_Q1[::2]], -0.1*np.ones(2), 'ro')
            ax.plot([-1,1], [-0.1,-0.1], 'b-')
            ax.axis("equal")
            pyplot.draw()
            pyplot.pause(1e-3)
        if step % solp.stride == 0:
            filename = "MCL-output/{:04}.npz".format(step)
            np.savez(filename, y_k=y_k, id_k=id_k, w_k=w_k, m3=m3, bmm=mesh.coord_map)
            print(Fore.GREEN + "{:>10}{:>16}{:>16}".format("Time", "Int. Disp.", "Sh. Disp.") + Style.RESET_ALL)
        if t >= solp.Te:
            print(Fore.GREEN + "Completed." + Style.RESET_ALL)
            break

        # =================================================================
        # Step 1. Update the reference contact line. 

        # extract the current reference CL locations
        chi_k = id_k.view(np.ndarray)[cl_dof_Q1[::2]] # (2,), with [0] being the left CL
        chi = np.zeros_like(chi_k)
        # ensure [0] is for the left CL; otherwise the code below does not make sense. 
        assert chi_k[0] < chi_k[1]
        assert M3_sp.dof_loc[0,0] < M3_sp.dof_loc[2,0]
        
        # project the discontinuous deformation gradient onto P1 to find the conormal vector m1
        dA_k = Measure(s_mesh, dim=1, order=5, coord_map=id_k) # the reference sheet mesh at the last time step
        q_P1_k_basis = FunctionBasis(Q_P1_sp, dA_k)
        C_L2 = c_L2.assemble(q_P1_k_basis, q_P1_k_basis, dA_k)
        L_DQ = l_dq.assemble(q_P1_k_basis, dA_k, q_k = q_k._interpolate(dA_k))
        dq_k = Function(Q_P1_sp)
        dq_k[:] = spsolve(C_L2, L_DQ)

        # extract the conormal at the contact line
        dq_k_at_cl = dq_k.view(np.ndarray)[cl_dof_Q1].reshape(-1, 2) # (-1, 2)
        m1_k = dq_k_at_cl / np.linalg.norm(dq_k_at_cl, ord=None, axis=1, keepdims=True) # (2, 2)
        m1_k[0] = -m1_k[0] 
        # find the displacement of the reference CL driven by unbalanced Young force
        a = np.sum(dq_k_at_cl * m1_k, axis=1) # (2, )
        m3_ = m3.view(np.ndarray).reshape(2,2)
        rcl_disp = - solp.dt / (phyp.mu_cl * a) * (phyp.gamma_1-phyp.gamma_2 + phyp.gamma_3 * np.sum(m3_ * m1_k, axis=1))
        chi = chi_k + rcl_disp

        # =================================================================
        # Step 2. Find the sheet mesh displacement. 

        xx = id_k.view(np.ndarray)[::2] # extract the x component of the mesh nodes
        s_mesh_disp = np.where(
            xx <= chi_k[0], (xx + 1.0) / (chi_k[0] + 1.0) * rcl_disp[0], 
            np.where(xx >= chi_k[1], (1.0 - xx) / (1.0 - chi_k[1]) * rcl_disp[1], 
                    (rcl_disp[0] * (chi_k[1] - xx) + rcl_disp[1] * (xx - chi_k[0])) / (chi_k[1] - chi_k[0]))
        )
        eta = Function(s_mesh.coord_fe) # the mesh velocity 
        eta[::2] = s_mesh_disp / solp.dt
        id = s_mesh.coord_map # type: Function
        id[::2] += s_mesh_disp # update the reference sheet mesh
        id_lift = lift_to_P2(Q_sp, id)

        # =================================================================
        # Step 3. Solve the fluid, the fluid-fluid interface, and the sheet deformation. 
        
        # set up the measures
        dx = Measure(mesh, dim=2, order=3)
        ds_i = Measure(mesh, dim=1, order=3, tags=(3,)) # the fluid interface restricted from the bulk mesh
        ds = Measure(i_mesh, dim=1, order=3)
        da_x = Measure(mesh, dim=1, order=5, tags=(4, 5)) # the deformed sheet restricted from the bulk mesh

        da_k = Measure(s_mesh, dim=1, order=5, coord_map=q_k) # the deformed sheet surface measure
        # dA_k = Measure(s_mesh, dim=1, order=5, coord_map=id_k) # the reference sheet mesh at the last time step; declared already
        dA = Measure(s_mesh, dim=1, order=5) # the reference sheet surface measure at the current time step

        dp_i = Measure(i_mesh, dim=0, order=1, tags=(9,)) # the CL restricted from the fluid interfacef
        dp_s = Measure(s_mesh, dim=0, order=1, tags=(9,)) # the CL on the reference sheet mesh
        dp_is = Measure(s_mesh, dim=0, order=1, tags=(9,), interiorFacet=True)
        dp = Measure(cl_mesh, dim=0, order=1)

        # =================================================================
        # set up the function bases:
        # fluid domain
        u_basis = FunctionBasis(U_sp, dx)
        p1_basis = FunctionBasis(P1_sp, dx)
        p0_basis = FunctionBasis(P0_sp, dx)
        # interface domain
        y_basis = FunctionBasis(Y_sp, ds)
        u_i_basis = FunctionBasis(U_sp, ds_i)
        k_basis = FunctionBasis(K_sp, ds)
        # boundary of interface (CL)
        y_cl_basis = FunctionBasis(Y_sp, dp_i)
        # sheet
        u_b_basis = FunctionBasis(U_sp, da_x)
        q_surf_basis = FunctionBasis(Q_sp, da_k)
        q_basis = FunctionBasis(Q_sp, dA)
        mom_basis = FunctionBasis(MOM_sp, dA)
        q_k_basis = FunctionBasis(Q_sp, dA_k) # also the basis for tau
        # CL from sheet
        q_cl_basis = FunctionBasis(Q_sp, dp_s)
        q_icl_basis = FunctionBasis(Q_sp, dp_is)
        mom_cl_basis = FunctionBasis(MOM_sp, dp_is)
        # CL
        m3_basis = FunctionBasis(M3_sp, dp)
        
        # =================================================================
        # assemble by blocks
        A_XIU = a_xiu.assemble(u_basis, u_basis, dx, eta=viscosity)
        A_XIP1 = a_xip.assemble(u_basis, p1_basis, dx)
        A_XIP0 = a_xip.assemble(u_basis, p0_basis, dx)
        A_XIK = a_xik.assemble(u_i_basis, k_basis, ds_i)
        A_XITAU = a_xitau.assemble(u_b_basis, q_surf_basis, da_x)

        A_ZY = a_zy.assemble(y_basis, y_basis, ds)
        A_ZK = a_zk.assemble(y_basis, k_basis, ds)
        A_ZM3 = a_zm3.assemble(y_cl_basis, m3_basis, dp_i)
        
        A_M3Q = a_zm3.assemble(m3_basis, q_cl_basis, dp_s)
        L_M3 = l_m3.assemble(m3_basis, dp_s, id=id._interpolate(dp_s))

        B_PIQ = c_L2.assemble(q_k_basis, q_k_basis, dA_k)
        B_PIU = c_L2.assemble(q_k_basis, u_b_basis, dA_k)
        # Note: u_b_basis is not on dA_k.
        # It is still OK as we do not need the gradient. 
        B_PITAU = b_pitau.assemble(q_k_basis, q_k_basis, dA_k, \
                                q_k = q_k._interpolate(dA_k), mu_i=slip_fric)
        L_PI_ADV = l_pi_adv.assemble(q_k_basis, dA_k, q_k = q_k._interpolate(dA_k), eta_k = eta._interpolate(dA_k))
        L_PI_Q = l_pi_L2.assemble(q_k_basis, dA_k, q_k=(q_k-id_lift)._interpolate(dA_k))

        #C_PHITAU = B_PIQ.T
        #C_PHIM3 = A_M3Q.T
        C_CL_PHIM = c_cl_phim.assemble(q_icl_basis, mom_cl_basis, dp_is)
        C_PHIM = c_phim.assemble(q_basis, mom_basis, dA)
        C_PHIQ = c_phiq.assemble(q_basis, q_basis, dA, w_k = w_k._interpolate(dA))
        C_PHIQ_SURF = c_phiq_surf.assemble(q_surf_basis, q_surf_basis, da_k, gamma=surf_tens)
        # C_CL_WQ = C_CL_PHIM.T
        # C_WQ = C_PHIM.T
        C_WM = c_L2.assemble(mom_basis, mom_basis, dA)

        # collect the block matrices
        #    u,        p1,      p0,      tau,      y,       k,      m3,        w,      m
        A = bmat((
            (A_XIU,    -A_XIP1, -A_XIP0, -A_XITAU, None,    -phyp.gamma_3*A_XIK, None, None,   None),  # u
            (A_XIP1.T, None,    None,    None,     None,    None,   None,      None,   None),  # p1
            (A_XIP0.T, None,    None,    None,     None,    None,   None,      None,   None),  # p0
            (-solp.dt*B_PIU, None, None, solp.dt*B_PITAU,   None, None, None,  B_PIQ,  None),  # tau
            (None,     None,    None,    None,     A_ZY,    A_ZK,   -A_ZM3,    None,   None),  # y
            (-solp.dt*A_XIK.T,  None, None, None,  A_ZK.T,  None,   None,      None,   None),  # k
            (None,     None,    None,    None,     A_ZM3.T, None,   None,      -A_M3Q, None),  # m3
            (None,     None,    None,    B_PIQ.T,  None,    None,   -phyp.gamma_3*A_M3Q.T, -phyp.Y*C_PHIQ-C_PHIQ_SURF, C_PHIM-C_CL_PHIM), # w
            (None,     None,    None,    None,     None,    None,   None,      C_PHIM.T-C_CL_PHIM.T, 1.0/phyp.B*C_WM), # m
        ), format="csr")
        # collect the right-hand-side
        u[:] = 0.0; p1[:] = 0.0; p0[:] = 0.0
        tau[:] = solp.dt * L_PI_ADV + L_PI_Q
        y[:] = 0.0; kappa[:] = A_ZK.T @ y_k
        m3[:] = L_M3
        w[:] = C_PHIQ_SURF @ id_lift
        mom[:] = 0.0
        L = group_fn(u, p1, p0, tau, y, kappa, m3, w, mom)

        # build the essential boundary conditions
        u_noslip_dof = np.unique(U_sp.getFacetDof(tags=(7,)))
        p_fix_dof = np.array((0,))
        # q_clamp_dof = np.unique(np.concatenate((Q_sp.getFacetDof(tags=(10,)), np.arange(1, Q_sp.num_dof, 2))))
        q_clamp_dof = np.unique(np.concatenate((Q_sp.getFacetDof(tags=(10,)).reshape(-1), np.arange(1, Q_sp.num_dof, 2)))) # for no-bending
        # mom_fix_dof = np.unique(MOM_sp.getFacetDof(tags=(10,)))
        mom_fix_dof = np.arange(MOM_sp.num_dof) # for no-bending
        free_dof = group_dof(
            (U_sp, P1_sp, P0_sp, Q_sp, Y_sp, K_sp, M3_sp, Q_sp, MOM_sp), 
            (u_noslip_dof, p_fix_dof, p_fix_dof, None, None, None, None, q_clamp_dof, mom_fix_dof)
        )
        # the essential boundary conditions are all homogeneous, 
        # so no need to homogeneize the right-hand-side.

        # solve the coupled system
        sol_free = spsolve(A[free_dof][:,free_dof], L[free_dof])
        sol_full = np.zeros_like(L)
        sol_full[free_dof] = sol_free
        split_fn(sol_full, u, p1, p0, tau, y, kappa, m3, w, mom)
        q = w + id_lift
        
        # =================================================================
        # Step 4. Displace the bulk mesh and update all the meshes. 

        # update the interface mesh
        i_mesh.coord_map[:] = y

        # no need to update the CL mesh as the coordinates are never used

        # the sheet mesh is already update in Step 2

        # solve for the bulk mesh displacement
        BMM_basis = FunctionBasis(mesh.coord_fe, dx) # for bulk mesh mapping
        A_EL = a_el.assemble(BMM_basis, BMM_basis, dx)
        # attach the displacement of the interface and the sheet
        BMM_int_dof = np.unique(mesh.coord_fe.getFacetDof(tags=(3,)))
        BMM_s_dof = np.unique(mesh.coord_fe.getFacetDof(tags=(4,5)))
        BMM_fix_dof = np.concatenate(mesh.coord_fe.getFacetDof(tags=(3,4,5,6,7,8)))
        bulk_disp = Function(mesh.coord_fe)
        bulk_disp[BMM_int_dof] = y - y_k
        bulk_disp[BMM_s_dof] = down_to_P1(Q_P1_sp, q - q_k)
        L_EL = Function(mesh.coord_fe)
        L_EL[:] = -A_EL @ bulk_disp # homogeneize the boundary conditions

        el_free_dof = group_dof((mesh.coord_fe,), (BMM_fix_dof,))
        bulk_disp[el_free_dof] = spsolve(A_EL[el_free_dof][:,el_free_dof], L_EL[el_free_dof])
        mesh.coord_map += bulk_disp

        print("{:>10.4f}{:>16.2e}{:>16.2e}".format(
            t + solp.dt, np.linalg.norm(y-y_k, ord=np.inf), np.linalg.norm(q-q_k, ord=np.inf)), end="")
        print("  ({:.4f}, {:.4f}), ".format(q[cl_dof_Q2[0]], q[cl_dof_Q2[2]]), end="")
        print("  ({:.4f}, {:.4f})".format(np.sqrt(m3[0]**2+m3[1]**2), np.sqrt(m3[2]**2+m3[3]**2)))

        step += 1

    # end time loop
    if solp.vis:
        pyplot.ioff()
        pyplot.show()
