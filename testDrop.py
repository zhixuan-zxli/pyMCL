from dataclasses import dataclass
import pickle
import numpy as np
from math import cos
from runner import *
from fem import *
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot
from matplotlib.collections import LineCollection
from colorama import Fore, Style

@dataclass
class PhysicalParameters:
    eta_2: float = 0.1
    mu_1: float = 1e1
    mu_2: float = 1e1
    mu_cl: float = 1.
    gamma_1: float = 0.
    gamma_3: float = 10.0
    gamma_2: float = 0. + 10.0 * cos(np.pi/2) # to be consistent: gamma_2 = gamma_1 + gamma_3 * cos(theta_Y)
    Cb: float = 1e-2
    Cs: float = 1e2
    pre: float = 0.2 # the initial Jacobian is 1 + pre

# ===========================================================
# functionals for calculating the energy
@Functional
def dx(x: QuadData) -> np.ndarray:
    return x.dx

# stretching energy
@Functional
def e_stretch(xi: QuadData, q_m: QuadData) -> np.ndarray:
    # q_m.grad: (2, 2, Ne, Nq)
    nu = (np.sum(q_m.grad[:,0]**2, axis=0, keepdims=True) - 1) / 2 # (1, Ne, Nq)
    return 0.5 * nu**2 * xi.dx

@Functional
def e_L2(x: QuadData, kappa: QuadData) -> np.ndarray:
    return 0.5 * np.sum(kappa**2, axis=0, keepdims=True) * x.dx

# ===========================================================
# forms for the sheet

@BilinearForm
def a_L2(w: QuadData, q: QuadData, x: QuadData, _) -> np.ndarray:
    return np.sum(w * q, axis=0, keepdims=True) * x.dx

@LinearForm
def l_dq(w: QuadData, x: QuadData, q_m: QuadData) -> np.ndarray:
    return np.sum(w * q_m.grad[:,0], axis=0, keepdims=True) * x.dx

# For a_etaq, use a_pir
# For a_etak, use a_L2

@BilinearForm
def a_xikappa(xi: QuadData, kappa: QuadData, x: QuadData, _) -> np.ndarray:
    tau = np.array((-x.cn[1], x.cn[0])) # (2, Ne, Nq)
    r1 = np.sum(xi.grad * kappa.grad, axis=(0,1)) # (Ne, Nq)
    r2 = np.sum(np.sum(xi.grad * tau[:,np.newaxis], axis=0) * np.sum(kappa.grad * tau[:,np.newaxis], axis=0), axis=0) # (Ne, Nq)
    r3 = (xi.grad[0,0] + xi.grad[1,1]) * (kappa.grad[0,0] + kappa.grad[1,1])
    return (r1 - 2*r2 + r3/2)[np.newaxis] * x.dx
# this comes from the linearization of the Willmore variation
@BilinearForm
def a_xiq_3(xi: QuadData, q: QuadData, x: QuadData, _, k_m: QuadData) -> np.ndarray:
    r1 = np.sum(k_m.grad * q.grad, axis=(0,1)) * (xi.grad[0,0] + xi.grad[1,1]) # (Ne, Nq)
    r2 = np.sum(xi.grad * q.grad, axis=(0,1)) * (k_m.grad[0,0] + k_m.grad[1,1]) # (Ne, Nq)
    return (r1 + r2)[np.newaxis] * x.dx

@LinearForm
def l_xi(xi: QuadData, x: QuadData, gamma: np.ndarray) -> np.ndarray:
    # gamma: (Nf, )
    return (gamma[:,np.newaxis] * (xi.grad[0,0] + xi.grad[1,1]))[np.newaxis] * x.dx

# this is for the stretching energy 
@BilinearForm
def a_xiq(xi: QuadData, q: QuadData, x: QuadData, _, q_m: QuadData) -> np.ndarray:
    # q_m.grad: (2, 2, Ne, Nq)
    nu = (np.sum(q_m.grad[:,0]**2, axis=0, keepdims=True) - 1) / 2 # (1, Ne, Nq)
    return nu * np.sum(q.grad[:,0] * xi.grad[:,0], axis=0, keepdims=True) * x.dx
# this comes from the linearization of the stretching energy
@BilinearForm
def a_xiq_2(xi: QuadData, q: QuadData, x: QuadData, _, q_m: QuadData) -> np.ndarray:
    # q_m.grad: (2, 2, Ne, Nq)
    return np.sum(q_m.grad[:,0]*q.grad[:,0], axis=0, keepdims=True) * np.sum(q_m.grad[:,0]*xi.grad[:,0], axis=0, keepdims=True) * x.dx

# for a_xim3, just use a_pim3

# the contact line condition
@BilinearForm
def a_m3m3(beta: QuadData, m3: QuadData, x: QuadData, _, m1_hat: QuadData) -> np.ndarray:
    # beta: (2, 1, 1)
    # m3: (2, 1, 1)
    return np.sum(beta*m1_hat, axis=0, keepdims=True) * np.sum(m3*m1_hat, axis=0, keepdims=True) * x.dx

@LinearForm
def l_m3(beta: QuadData, x: QuadData, m1_hat: QuadData) -> np.ndarray:
    return np.sum(beta*m1_hat, axis=0, keepdims=True) * x.dx

# ===========================================================
# bilinear forms for the fluid and fluid interface

@BilinearForm
def a_vu(v: QuadData, u: QuadData, x: QuadData, _, eta) -> np.ndarray:
    # eta: (Ne,)
    # grad: (2, 2, Ne, Nq)    
    z = np.zeros(x.shape[1:]) # (Ne, Nq)
    for i, j in (0,0), (0,1), (1,0), (1,1):
        z += (u.grad[i,j] + u.grad[j,i]) * v.grad[i,j]
    return (z * eta[:, np.newaxis])[np.newaxis] * x.dx

@BilinearForm
def a_vp(v: QuadData, p: QuadData, x: QuadData, _) -> np.ndarray:
    # v.grad: (2,2,Ne,Nq)
    # p: (1, 1, Nq)
    z = (v.grad[0,0] + v.grad[1,1]) * p[0]
    return z[np.newaxis] * x.dx

@BilinearForm
def a_vu_nitsche(v: QuadData, u: QuadData, x: QuadData, _, eta: np.ndarray, mu: np.ndarray, alpha: float, x2: QuadData) -> np.ndarray:
    # eta, mu: (Nf, )
    # alpha is the stabilization factor
    assert x2.shape == x.shape
    tau = np.array((-x2.cn[1], x2.cn[0]))
    n_Tu_n = 2 * eta[:, np.newaxis] * np.sum(np.sum(u.grad * x2.cn[np.newaxis], axis=1) * x2.cn, axis=0) # (Nf, Nq)
    n_Tv_n = 2 * eta[:, np.newaxis] * np.sum(np.sum(v.grad * x2.cn[np.newaxis], axis=1) * x2.cn, axis=0) # (Nf, Nq)
    u_n = np.sum(u * x2.cn, axis=0) # (Nf, Nq)
    u_t = np.sum(u * tau, axis=0) # (Nf, Nq)
    v_n = np.sum(v * x2.cn, axis=0) # (Nf, Nq)
    v_t = np.sum(v * tau, axis=0) # (Nf, Nq)
    return (-n_Tu_n * v_n + n_Tv_n * u_n + mu[:, np.newaxis] * u_t * v_t)[np.newaxis] * x2.dx \
        + (alpha * u_n * v_n)[np.newaxis]

@BilinearForm
def a_vp_nitsche(v: QuadData, p: QuadData, x: QuadData, _, x2: QuadData) -> np.ndarray:
    assert x.shape == x2.shape
    v_n = np.sum(v * x2.cn, axis=0) # (Nf, Nq)
    return (p[0] * v_n)[np.newaxis] * x2.dx

@BilinearForm
def a_vq_nitsche_n(v: QuadData, q: QuadData, _, x: QuadData, eta: np.ndarray) -> np.ndarray:
    # q: (2, Nf, Nq)
    # x.cn: (2, Nf, Nq)
    n_Tv_n = 2 * eta[:, np.newaxis] * np.sum(np.sum(v.grad * x.cn[np.newaxis], axis=1) * x.cn, axis=0) # (Nf, Nq)
    q_n = np.sum(q * x.cn, axis=0) # (Nf, Nq)
    return n_Tv_n*q_n[np.newaxis] * x.dx

@BilinearForm
def a_vq_nitsche_t(v: QuadData, q: QuadData, _, x: QuadData, mu: np.ndarray) -> np.ndarray:
    # q: (2, Nf, Nq)
    # x.cn: (2, Nf, Nq)
    tau = np.array((-x.cn[1], x.cn[0]))
    v_t = np.sum(v * tau, axis=0)
    q_t = np.sum(q * tau, axis=0) # (Nf, Nq)
    return (mu[:, np.newaxis]*v_t*q_t)[np.newaxis] * x.dx

@BilinearForm
def a_vq_nitsche_s(v: QuadData, q: QuadData, _, x: QuadData, alpha: float) -> np.ndarray:
    # q: (2, Nf, Nq)
    # x.cn: (2, Nf, Nq)
    v_n = np.sum(v * x.cn, axis=0)
    q_n = np.sum(q * x.cn, axis=0) # (Nf, Nq)
    return (alpha * q_n * v_n)[np.newaxis]

@BilinearForm
def a_rhoq_nitsche(rho: QuadData, q: QuadData, _, x: QuadData) -> np.ndarray:
    q_n = np.sum(q * x.cn, axis=0)
    return (rho[0] * q_n)[np.newaxis] * x.dx

@BilinearForm
def a_vomg(v: QuadData, omega: QuadData, x: QuadData, _) -> np.ndarray:
    # omega: (1, 1, Nq)
    # v: (2, Nf, Nq)
    # x.fn: (2, Nf, Nq)
    return np.sum(x.fn * v, axis=0, keepdims=True) * omega * x.ds # (2, Nf, Nq)

@BilinearForm
def a_pir(pi: QuadData, r: QuadData, x: QuadData, _) -> np.ndarray:
    return np.sum(pi.grad * r.grad, axis=(0,1))[np.newaxis] * x.dx

@BilinearForm
def a_piomg(pi: QuadData, omega: QuadData, x: QuadData, _) -> np.ndarray:
    return np.sum(pi * x.cn, axis=0, keepdims=True) * omega * x.dx

@BilinearForm
def a_pim3(pi: QuadData, m3: QuadData, x: QuadData, _) -> np.ndarray:
    # z, m3: (2, Nf, Nq=1)
    return np.sum(pi * m3, axis=0, keepdims=True) * x.ds

# ===========================================================
# linear elasticity for the bulk mesh displacement
@BilinearForm
def a_el(Z: QuadData, Y: QuadData, x: QuadData, _) -> np.ndarray:
    # grad: (2, 2, Ne, Nq)
    # x.dx: (1, Ne, Nq)
    lam_dx = x.dx + (x.dx.max() - x.dx.min()) # (1, Ne, Nq)
    r = np.zeros(x.shape[1:]) # (Ne,Nq)
    for i, j in (0,0), (0,1), (1,0), (1,1):
        r += (Y.grad[i,j] + Y.grad[j,i]) * Z.grad[i,j]
    r += (Y.grad[0,0] + Y.grad[1,1]) * (Z.grad[0,0] + Z.grad[1,1])
    return r[np.newaxis] * lam_dx

# ===========================================================
# some helper functions

def lift_to_P2(P2_space: FunctionSpace, p1_func: Function) -> Function:
    p2_func = Function(P2_space)
    p2_func[:p1_func.size] = p1_func
    rdim = P2_space.elem.rdim
    assert rdim == p1_func.fe.elem.rdim
    if P2_space.mesh.tdim == 1:
        for d in range(rdim):
            p2_func[P2_space.elem_dof[2*rdim+d]] = 0.5 * \
                (p1_func[P2_space.elem_dof[d]] + p1_func[P2_space.elem_dof[rdim+d]])
    else:
        raise NotImplementedError
    return p2_func

def down_to_P1(P1_space: FunctionSpace, p2_func: Function) -> Function:
    assert P1_space.mesh.tdim == 1
    assert P1_space.elem.rdim == p2_func.fe.elem.rdim
    p1_func = Function(P1_space)
    p1_func[:] = p2_func[:P1_space.num_dof]
    return p1_func

# arrange the finite element vector as a finite difference vector, 
# i.e. the nodes are arranged from left to right in space.
def arrange_as_FD(P2_space: FunctionSpace, p2_func: Function) -> np.ndarray:
    assert P2_space.mesh.tdim == 1
    rdim = P2_space.elem.rdim
    arr = np.zeros(p2_func.shape).reshape(-1, rdim)
    for d in range(rdim):
        arr[0:-1:rdim, d] = p2_func[P2_space.elem_dof[d]]
        arr[-1, d] = p2_func[P2_space.elem_dof[rdim+d, -1]]
        arr[1::rdim, d] = p2_func[P2_space.elem_dof[2*rdim+d]]
    return arr

# arrange a finite difference vector as a finite element vector. 
def arrange_as_FE(P2_space: FunctionSpace, arr: np.ndarray) -> Function:
    assert P2_space.mesh.tdim == 1
    rdim = P2_space.elem.rdim
    func = Function(P2_space)
    for d in range(rdim):
        func[P2_space.elem_dof[d]] = arr[0:-1:rdim, d]
        func[P2_space.elem_dof[rdim+d, -1]] = arr[-1, d]
        func[P2_space.elem_dof[2*rdim+d]] = arr[1::rdim, d]
    return func


# ===========================================================

class Drop_Runner(Runner):

    def prepare(self) -> None:
        super().prepare()
        self.phyp = PhysicalParameters()
        with open(self._get_output_name("PhysicalParameters"), "wb") as f:
            pickle.dump(self.phyp, f)
        
        # load the bulk mesh and build the submeshes
        # physical groups from GMSH
        # group_name = {"fluid_1": 1, "fluid_2": 2, "interface": 3, "dry": 4, "wet": 5, \
        #              "right": 6, "top": 7, "sym": 8, "cl": 9, "clamp": 10, "isym": 11, "s_sym": 12}
        self.mesh = Mesh()
        self.mesh.load(self.args.mesh_name)
        for _ in range(self.args.spaceref):
            self.mesh = splitRefine(self.mesh)
        setMeshMapping(self.mesh)
        self.i_mesh = self.mesh.view(1, tags=(3, )) # interface mesh
        setMeshMapping(self.i_mesh)
        self.s_mesh = self.mesh.view(1, tags=(4, 5)) # sheet reference mesh
        setMeshMapping(self.s_mesh)
        self.cl_mesh = self.mesh.view(0, tags=(9, )) # contact line mesh
        setMeshMapping(self.cl_mesh)

        # prepare the variable coefficients
        self.viscosity = np.where(self.mesh.cell_tag[2] == 1, 1.0, self.phyp.eta_2)
        self.viscosity_bound = np.where(self.s_mesh.cell_tag[1] == 5, 1.0, self.phyp.eta_2)
        self.slip_fric = np.where(self.s_mesh.cell_tag[1] == 5, self.phyp.mu_1, self.phyp.mu_2)
        self.surf_tens = np.where(self.s_mesh.cell_tag[1] == 5, self.phyp.gamma_1, self.phyp.gamma_2)

        # set up the function spaces
        self.U_sp = FunctionSpace(self.mesh, VectorElement(TriP2, 2))
        self.P1_sp = FunctionSpace(self.mesh, TriP1)
        self.P0_sp = FunctionSpace(self.mesh, TriDG0)
        self.R_sp = self.i_mesh.coord_fe # type: FunctionSpace # should be VectorElement(LineP1, 2)
        self.OMG_sp = FunctionSpace(self.i_mesh, LineP1)
        self.Q_sp = FunctionSpace(self.s_mesh, VectorElement(LineP2, 2)) # for the sheet deformation and the mean curvature vector
        self.Q_P1_sp = self.s_mesh.coord_fe # type: FunctionSpace
        self.M3_sp = FunctionSpace(self.cl_mesh, VectorElement(NodeElement, 2))
        # assert self.M3_sp.dof_loc[0,0] < self.M3_sp.dof_loc[2,0]

        # extract the DOFs for the bulk mesh mapping
        Y_sp = self.mesh.coord_fe # type: FunctionSpace
        self.BMM_int_dof = Y_sp.getDofByLocation(self.R_sp.dof_loc[::2])
        self.BMM_sh_dof = Y_sp.getDofByLocation(self.Q_P1_sp.dof_loc[::2])
        self.BMM_sym_dof = np.where(Y_sp.dof_loc[:,0] < 1e-12)[0]
        self.BMM_bound_dof = np.where((Y_sp.dof_loc[:,0] > 1-1e-12) | (Y_sp.dof_loc[:,1] > 1-1e-12))[0]
        BMM_fix_dof = np.unique(np.concatenate((self.BMM_int_dof, self.BMM_sh_dof, self.BMM_bound_dof, self.BMM_sym_dof[::2])))
        self.BMM_free_dof = group_dof((Y_sp,), (BMM_fix_dof,))

        # extract the DOFs at the contact line
        cl_pos = self.cl_mesh.point.copy() # shape (1, 2)
        self.cl_dof_R = np.where(np.linalg.norm(self.R_sp.dof_loc - cl_pos, axis=1) < 1e-12)[0]
        assert self.cl_dof_R.size == 2 
        self.cl_dof_Q = np.where(np.linalg.norm(self.Q_sp.dof_loc - cl_pos, axis=1) < 1e-12)[0]
        assert self.cl_dof_Q.size == 2 
        self.cl_dof_Q_P1 = np.where(np.linalg.norm(self.Q_P1_sp.dof_loc - cl_pos, axis=1) < 1e-12)[0]
        assert self.cl_dof_Q_P1.size == 2

        # extract the useful DOFs for the velocity and the sheet deformation
        u_noslip_dof = np.where(self.U_sp.dof_loc[:,1] > 1-1e-12)[0]
        u_sym_dof = np.where(self.U_sp.dof_loc[:,0] < 1e-12)[0]
        # u_bot_dof = np.where(self.U_sp.dof_loc[:,1] < 1e-12)[0]
        u_fix_dof = np.unique(np.concatenate((u_noslip_dof, u_sym_dof[::2])))
        p_fix_dof = np.array((0,), dtype=np.int32) # np.arange(self.P0_sp.num_dof, dtype=np.int32)
        r_sym_dof = np.where(self.R_sp.dof_loc[:,0] < 1e-12)[0] # should be (2, )
        self.q_clamp_dof = np.where(self.Q_sp.dof_loc[:,0] > 1-1e-12)[0]
        self.q_sym_dof = np.where(self.Q_sp.dof_loc[:,0] < 1e-12)[0]
        # q_all_dof = np.arange(self.Q_sp.num_dof)
        q_fix_dof = np.unique(np.concatenate((self.q_clamp_dof, self.q_sym_dof[0:1])))
        self.free_dof = group_dof(
            (self.U_sp, self.P1_sp, self.P0_sp, self.R_sp, self.OMG_sp, self.Q_sp, self.Q_sp, self.M3_sp), 
            (u_fix_dof, None, p_fix_dof, r_sym_dof[0], None, q_fix_dof, q_fix_dof, None) 
        )
        print("Number of free dofs = {}".format(self.free_dof.sum()))

        # allocate the solution functions
        self.u = Function(self.U_sp)
        self.p1 = Function(self.P1_sp)
        self.p0 = Function(self.P0_sp)
        self.r = Function(self.R_sp)
        self.r_m = Function(self.R_sp)
        self.omega = Function(self.OMG_sp)
        self.q = None # the deformation map
        self.q_m = Function(self.Q_sp) # the deformation map
        self.dqdt = Function(self.Q_sp) # the velocity of the deformation map
        self.kappa = Function(self.Q_sp) # the mean curvature vector
        self.k_m = Function(self.Q_sp)
        self.m3 = Function(self.M3_sp)
        self.id_m = Function(self.s_mesh.coord_fe) # the mesh mapping of the reference sheet
        self.m1_hat = Function(self.M3_sp)

        # redefine the mesh mapping for the sheet to fulfill the pre-stretch condition
        self.q = lift_to_P2(self.Q_sp, self.s_mesh.coord_map)
        self.s_mesh.coord_map[:] *= 1. / (1. + self.phyp.pre)
        
        # mark the CL nodes for finite difference arrangement
        _temp = np.zeros_like(self.q, dtype=np.int_)
        _temp[self.cl_dof_Q] = (1, 0)
        self.cl_dof_fd = np.where(arrange_as_FD(self.Q_sp, _temp)[:,0])[0] # shape (1, )

        # initialize the arrays for storing the energy history
        self.energy = np.zeros((self.num_steps+1, 5))
        # ^ the columns are stretching energy, bending energy, surface energy for Sigma_1, 2, 3. 
        self.phycl_hist = np.zeros((self.num_steps+1, 2)) # history of physical CL
        self.refcl_hist = np.zeros((self.num_steps+1, 2)) # history of reference CL

        # read checkpoints from file
        if self.args.resume:
            self.mesh.coord_map[:] = self.resume_file["bulk_coord_map"]
            self.i_mesh.coord_map[:] = self.resume_file["r_m"]
            self.s_mesh.coord_map[:] = self.resume_file["id_m"]
            self.q[:] = self.resume_file["q_m"]
            self.kappa[:] = self.resume_file["k_m"]
            self.energy[:self.step+1] = self.resume_file["energy"]
            self.phycl_hist[:self.step+1] = self.resume_file["phycl_hist"]
            self.refcl_hist[:self.step+1] = self.resume_file["refcl_hist"]
            del self.resume_file

        # prepare visualization
        if self.args.vis:
            pyplot.ion()
            self.ax = pyplot.subplot()
            self.ax.axis("equal")
            self.bulk_triangles = self.mesh.coord_fe.elem_dof[::2,:].T//2

    def pre_step(self) -> bool:
        # retrieve the mesh mapping which will be needed in the measures. 
        self.r_m[:] = self.i_mesh.coord_map
        self.id_m[:] = self.s_mesh.coord_map
        self.q_m[:] = self.q
        self.q_m_down = down_to_P1(self.Q_P1_sp, self.q_m)
        self.k_m[:] = self.kappa

        # record the CL locations
        refcl = self.id_m.view(np.ndarray)[self.cl_dof_Q_P1]
        self.refcl_hist[self.step] = refcl
        phycl = self.q.view(np.ndarray)[self.cl_dof_Q]
        self.phycl_hist[self.step] = phycl

        # calculate the energy
        d_xi = Measure(self.s_mesh, dim=1, order=5, coord_map=self.id_m) # the reference sheet mesh at the last time step
        self.energy[self.step, 0] = self.phyp.Cs * e_stretch.assemble(d_xi, q_m=self.q_m._interpolate(d_xi))
        da = Measure(self.s_mesh, dim=1, order=5, coord_map=self.q_m) # the deformed sheet surface measure
        self.energy[self.step, 1] = self.phyp.Cb * e_L2.assemble(da, kappa=self.k_m._interpolate(da))
        da_1 = Measure(self.s_mesh, dim=1, order=5, tags=(5,), coord_map=self.q_m)
        self.energy[self.step, 2] = self.phyp.gamma_1 * dx.assemble(da_1)
        da_2 = Measure(self.s_mesh, dim=1, order=5, tags=(4,), coord_map=self.q_m)
        self.energy[self.step, 3] = self.phyp.gamma_2 * dx.assemble(da_2)
        ds = Measure(self.i_mesh, dim=1, order=3)
        self.energy[self.step, 4] = self.phyp.gamma_3 * dx.assemble(ds)
        print("energy={:.5f}, ".format(np.sum(self.energy[self.step])), end="")

        t = self.step * self.solp.dt
        if self.args.vis:
            self.ax.clear()
            # press = self.p0.view(np.ndarray)[self.P0_sp.elem_dof][0] + np.sum(self.p1.view(np.ndarray)[self.P1_sp.elem_dof], axis=0) / 3 # (Nt, )
            # tpc = self.ax.tripcolor(self.mesh.coord_map[::2], self.mesh.coord_map[1::2], press, triangles=self.bulk_triangles)
            # if not hasattr(self, "colorbar"):
            #     self.colorbar = pyplot.colorbar(tpc)
            # else:
            #     self.colorbar.update_normal(tpc)
            # self.ax.triplot(self.mesh.coord_map[::2], self.mesh.coord_map[1::2], triangles=self.bulk_triangles, linewidth=0.5)
            # plot the velocity
            _u = self.u.view(np.ndarray); _n = self.mesh.coord_map.size; 
            self.ax.quiver(self.mesh.coord_map[::2], self.mesh.coord_map[1::2], _u[:_n:2], _u[1:_n:2], color="tab:blue") #, _mag, cmap=pyplot.get_cmap("Spectral"))
            # plot the interface
            _r_m = self.r_m.view(np.ndarray)
            segments = _r_m[self.i_mesh.coord_fe.elem_dof].reshape(2, 2, -1).transpose(2, 0, 1)
            self.ax.add_collection(LineCollection(segments=segments, colors="tab:green"))
            # plot the sheet
            _q_m_down = self.q_m_down.view(np.ndarray)
            segments = _q_m_down[self.s_mesh.coord_fe.elem_dof].reshape(2, 2, -1).transpose(2, 0, 1)
            self.ax.add_collection(LineCollection(segments=segments, colors="tab:orange"))
            self.ax.plot(self.q_m[::2], self.q_m[1::2], "k+")
            self.ax.plot(self.q_m[::2], self.dqdt[::2], 'ro')
            self.ax.plot(self.q_m[::2], self.dqdt[1::2], 'bo')
            # plot the cornormals
            self.ax.quiver(phycl[0], phycl[1], self.m1_hat[0], self.m1_hat[1], color="tab:brown") # m1
            self.ax.quiver(phycl[0], phycl[1], self.m3[0], self.m3[1], color="tab:brown") # m3
            # plot the frame
            self.ax.plot((0.0, 1.0, 1.0), (1.0, 1.0, 0.0), 'k-')
            self.ax.set_xlim(0.0, 1.0); self.ax.set_ylim(-0.15, 1.0)
            pyplot.title("t={:.5f}".format(t))
            pyplot.draw()
            pyplot.pause(1e-4)
            # output image files
            if self.step % self.solp.stride_frame == 0:
                filename = self._get_output_name("{:05}.png".format(self.step))
                pyplot.savefig(filename, dpi=300.0)
        if self.step % self.solp.stride_checkpoint == 0:
            filename = self._get_output_name("{:05}.npz".format(self.step))
            np.savez(filename, bulk_coord_map=self.mesh.coord_map, r_m=self.r_m, id_m=self.id_m, q_m=self.q_m, k_m=self.kappa, 
                     phycl_hist=self.phycl_hist[:self.step+1], 
                     refcl_hist=self.refcl_hist[:self.step+1], 
                     energy=self.energy[:self.step+1])
            print(Fore.GREEN + "\nCheckpoint saved to " + filename + Style.RESET_ALL)
        
        return self.step >= self.num_steps
    
    def main_step(self) -> None:
        phyp = self.phyp # just for convenience

        # Step 0. Get an explicit estimate of m1. 
        q_fd = arrange_as_FD(self.Q_sp, self.q_m) # (n, 2)
        x, y = q_fd[:,0], q_fd[:,1]
        j = self.cl_dof_fd[0]
        dydx = (y[j]-y[j-1])*(1/(x[j]-x[j-1]) - 1/(x[j+1]-x[j-1])) + (y[j+1]-y[j])*(1/(x[j+1]-x[j]) - 1/(x[j+1]-x[j-1]))
        self.m1_hat[:] = (1.0, dydx)
        self.m1_hat /= np.linalg.norm(self.m1_hat)

        # =================================================================
        # Step 1. Solve the fluid, the fluid-fluid interface, and the sheet deformation. 
        
        # set up the measures
        dx = Measure(self.mesh, dim=2, order=3)
        ds_4u = Measure(self.mesh, dim=1, order=5, tags=(3,)) # the fluid interface restricted from the bulk mesh
        da_4u = Measure(self.mesh, dim=1, order=5, tags=(4,5)) # the deformed sheet restricted from the bulk mesh

        ds = Measure(self.i_mesh, dim=1, order=5)
        da = Measure(self.s_mesh, dim=1, order=5, coord_map=self.q_m) # the deformed sheet surface measure
        d_xi = Measure(self.s_mesh, dim=1, order=5, coord_map=self.id_m) # the reference sheet mesh at the last time step

        dp_i = Measure(self.i_mesh, dim=0, order=1, tags=(9,)) # the CL restricted from the fluid interface
        dp_s = Measure(self.s_mesh, dim=0, order=1, tags=(9,), coord_map=self.q_m) # the CL on the reference sheet mesh
        dp = Measure(self.cl_mesh, dim=0, order=1)

        # set up the function bases
        # fluid domain
        u_basis = FunctionBasis(self.U_sp, dx)
        p1_basis = FunctionBasis(self.P1_sp, dx)
        p0_basis = FunctionBasis(self.P0_sp, dx)
        # interface domain
        r_basis = FunctionBasis(self.R_sp, ds)
        u_ds_basis = FunctionBasis(self.U_sp, ds_4u)
        omega_basis = FunctionBasis(self.OMG_sp, ds)
        # sheet domain
        u_da_basis = FunctionBasis(self.U_sp, da_4u)
        p1_da_basis = FunctionBasis(self.P1_sp, da_4u)
        p0_da_basis = FunctionBasis(self.P0_sp, da_4u)
        q_da_basis = FunctionBasis(self.Q_sp, da)
        q_ref_basis = FunctionBasis(self.Q_sp, d_xi)
        # CL from sheet
        r_cl_basis = FunctionBasis(self.R_sp, dp_i)
        q_cl_basis = FunctionBasis(self.Q_sp, dp_s)
        m3_basis = FunctionBasis(self.M3_sp, dp)
        
        # assemble by blocks
        alpha_stab = 10.0 # the stablization constant in Nitsche method
        # for the Stokes equation
        A_VU = a_vu.assemble(u_basis, u_basis, None, None, eta=self.viscosity)
        A_VP1 = a_vp.assemble(u_basis, p1_basis)
        A_VP0 = a_vp.assemble(u_basis, p0_basis)
        # for the Nitsche method
        A_VU_NIT = a_vu_nitsche.assemble(u_da_basis, u_da_basis, None, None, \
                    eta = self.viscosity_bound, mu = self.slip_fric, alpha = alpha_stab, x2 = da.x)
        A_VP1_NIT = a_vp_nitsche.assemble(u_da_basis, p1_da_basis, x2=da.x)
        A_VP0_NIT = a_vp_nitsche.assemble(u_da_basis, p0_da_basis, x2=da.x)
        A_VQ_NIT_N = a_vq_nitsche_n.assemble(u_da_basis, q_da_basis, None, None, eta=self.viscosity_bound)
        A_VQ_NIT_T = a_vq_nitsche_t.assemble(u_da_basis, q_da_basis, None, None, mu=self.slip_fric)
        A_VQ_NIT_S = a_vq_nitsche_s.assemble(u_da_basis, q_da_basis, None, None, alpha=alpha_stab)
        A_RHO1Q_NIT = a_rhoq_nitsche.assemble(p1_da_basis, q_da_basis)
        A_RHO0Q_NIT = a_rhoq_nitsche.assemble(p0_da_basis, q_da_basis)
        # for the fluid interface
        A_VOMG = a_vomg.assemble(u_ds_basis, omega_basis)
        A_PIR = a_pir.assemble(r_basis, r_basis)
        A_PIOMG = a_piomg.assemble(r_basis, omega_basis)
        A_PIM3 = a_pim3.assemble(r_cl_basis, m3_basis)
        # for the compatibility of the sheet
        A_ETAQ = a_pir.assemble(q_da_basis, q_da_basis)
        A_ETAK = a_L2.assemble(q_da_basis, q_da_basis)
        # for the mechanics of the sheet
        A_XIK = a_xikappa.assemble(q_da_basis, q_da_basis)
        A_XIQ = a_xiq.assemble(q_ref_basis, q_ref_basis, None, None, q_m=self.q_m._interpolate(d_xi))
        A_XIQ_2 = a_xiq_2.assemble(q_ref_basis, q_ref_basis, None, None, q_m=self.q_m._interpolate(d_xi))
        A_XIQ_3 = a_xiq_3.assemble(q_da_basis, q_da_basis, None, None, k_m=self.k_m._interpolate(da))
        A_XIQ_T = a_vq_nitsche_t.assemble(q_da_basis, q_da_basis, None, None, mu=self.slip_fric)
        A_XIQ_S = a_vq_nitsche_s.assemble(q_da_basis, q_da_basis, None, None, alpha=alpha_stab)
        L_XI = l_xi.assemble(q_da_basis, None, gamma=self.surf_tens)
        A_XIM3 = a_pim3.assemble(q_cl_basis, m3_basis)
        # for the contact line condition
        A_M3M3 = a_m3m3.assemble(m3_basis, m3_basis, None, None, m1_hat=self.m1_hat._interpolate(dp))
        L_M3 = l_m3.assemble(m3_basis, None, m1_hat=self.m1_hat._interpolate(dp))

        # collect the block matrices
        dt = solp.dt
        #    u,             p1,               p0,               r,          omega,                q,                                 kappa,     m3
        A = bmat((
            (A_VU+A_VU_NIT, -A_VP1+A_VP1_NIT, -A_VP0+A_VP0_NIT, None,       -phyp.gamma_3*A_VOMG, -A_VQ_NIT_N-A_VQ_NIT_T-A_VQ_NIT_S, None,      None),                   # v
            (A_VP1.T-A_VP1_NIT.T,  None,      None,             None,       None,                 A_RHO1Q_NIT,                       None,      None),                   # rho_1
            (A_VP0.T-A_VP0_NIT.T,  None,      None,             None,       None,                 A_RHO0Q_NIT,                       None,      None),                   # rho_0
            (None,          None,             None,             A_PIR,      A_PIOMG,              None,                              None,      -A_PIM3),                # pi
            (-dt*A_VOMG.T,  None,             None,             A_PIOMG.T,  None,                 None,                              None,      None),                   # delta
            (-A_VQ_NIT_N.T-A_VQ_NIT_T.T-A_VQ_NIT_S.T, A_RHO1Q_NIT.T, A_RHO0Q_NIT.T, None, None,   -phyp.Cs*dt*(A_XIQ+A_XIQ_2)-2*phyp.Cb*dt*A_XIQ_3-A_XIQ_T-A_XIQ_S, phyp.Cb*A_XIK, -phyp.gamma_3*A_XIM3), # xi
            (None,          None,             None,             None,       None,                 dt*A_ETAQ,                         A_ETAK,    None),                   # eta
            (None,          None,             None,             phyp.mu_cl/dt*A_PIM3.T, None,     -phyp.mu_cl*A_XIM3.T,              None,      phyp.gamma_3*A_M3M3),    # m3
        ), format="csc")
        # collect the right-hand side
        self.u[:] = 0.0
        self.p1[:] = 0.0
        self.p0[:] = 0.0
        self.r[:] = 0.0
        self.omega[:] = A_PIOMG.T @ self.r_m
        self.q[:] = L_XI + phyp.Cs * A_XIQ @ self.q_m
        self.kappa[:] = -A_ETAQ @ self.q_m
        self.m3[:] = (phyp.gamma_2 - phyp.gamma_1) * L_M3 + phyp.mu_cl/dt*A_PIM3.T @ self.r_m
        L = group_fn(self.u, self.p1, self.p0, self.r, self.omega, self.q, self.kappa, self.m3)
        # set up the boundary conditions and homogeneize the right-hand side
        # self.omega[:] = 0.0
        # self.q[:] = 0.0
        # self.kappa[:] = 0.0
        # self.m3[:] = 0.0
        # sol_full = group_fn(self.u, self.p1, self.p0, self.r, self.omega, self.q, self.kappa, self.m3)
        sol_full = np.zeros_like(L)
        # L = L - A @ sol_full
        # solve the coupled system
        free_dof = self.free_dof
        sol_free = spsolve(A[free_dof][:,free_dof], L[free_dof])
        sol_full[free_dof] = sol_free
        split_fn(sol_full, self.u, self.p1, self.p0, self.r, self.omega, self.q, self.kappa, self.m3)

        # q is the deformation velocity, and update to get the deformation map
        np.copyto(self.dqdt, self.q)
        self.q[:] = self.q_m + dt * self.q
        
        # =================================================================
        # Step 2. Convert to FD form and find the derivatives using finite difference 
        q_fd = arrange_as_FD(self.Q_sp, self.q) # (n, 2)
        xi_fd = arrange_as_FD(self.Q_sp, lift_to_P2(self.Q_sp, self.s_mesh.coord_map))[:,0][:, np.newaxis] # (n, 1)
        assert np.all(xi_fd[1:,0] - xi_fd[:-1,0] > 0), "The mesh mapping is not strictly increasing!"
        assert np.all(q_fd[1:,0] - q_fd[:-1,0] > 0), "The mesh mapping is not strictly increasing!"
        ref_cl = xi_fd[self.cl_dof_fd] # (1,)
        # find the first derivative to the right and to the left
        dq_plus = np.zeros_like(q_fd)
        dq_plus[:-2] = (q_fd[1:-1] - q_fd[:-2]) * (1/(xi_fd[1:-1]-xi_fd[:-2]) + 1/(xi_fd[2:]-xi_fd[:-2])) + \
            (q_fd[2:] - q_fd[1:-1]) * (1/(xi_fd[2:]-xi_fd[:-2]) - 1/(xi_fd[2:]-xi_fd[1:-1]))
        dq_minus = np.zeros_like(q_fd)
        dq_minus[2:] = (q_fd[1:-1]-q_fd[:-2]) * (1/(xi_fd[2:]-xi_fd[:-2]) - 1/(xi_fd[1:-1]-xi_fd[:-2])) + \
            (q_fd[2:]-q_fd[1:-1]) * (1/(xi_fd[2:]-xi_fd[1:-1]) + 1/(xi_fd[2:]-xi_fd[:-2]))

        slip_cl = self.r.view(np.ndarray)[self.cl_dof_R] - self.q.view(np.ndarray)[self.cl_dof_Q] # type: np.ndarray # (2,)
        # find the reference CL velocity
        if slip_cl[0] > 0:
            d_chi = np.dot(slip_cl, dq_plus[self.cl_dof_fd[0]]) / np.sum(dq_plus[self.cl_dof_fd[0]]**2)
        else:
            d_chi = np.dot(slip_cl, dq_minus[self.cl_dof_fd[0]]) / np.sum(dq_minus[self.cl_dof_fd[0]]**2)
        # adjust the reference mesh mapping
        d_xi = np.where(
            xi_fd <= ref_cl[0], xi_fd / ref_cl[0] * d_chi, (xi_fd[-1]-xi_fd) / (xi_fd[-1]-ref_cl[0]) * d_chi
        ) # (n, ) # check!

        # advect the deformation map using a Lax-Wendroff scheme
        q_next = q_fd.copy()
        q_next[1:-1] += d_xi[1:-1] * ((q_fd[1:-1]-q_fd[:-2])*(1/(xi_fd[1:-1]-xi_fd[:-2])-1/(xi_fd[2:]-xi_fd[:-2])) + (q_fd[2:]-q_fd[1:-1])*(1/(xi_fd[2:]-xi_fd[1:-1])-1/(xi_fd[2:]-xi_fd[:-2])))
        q_next[1:-1] += d_xi[1:-1]**2 * ((q_fd[2:]-q_fd[1:-1])/(xi_fd[2:]-xi_fd[1:-1]) - (q_fd[1:-1]-q_fd[:-2])/(xi_fd[1:-1]-xi_fd[:-2])) / (xi_fd[2:]-xi_fd[:-2])
        q_next[self.cl_dof_fd] = self.r.view(np.ndarray)[self.cl_dof_R] # check it!
        
        # now overwrite the mesh mapping and the deformation
        self.s_mesh.coord_map += down_to_P1(self.Q_P1_sp, arrange_as_FE(self.Q_sp, np.concatenate((d_xi, np.zeros_like(d_xi)), axis=1)))
        self.q[:] = arrange_as_FE(self.Q_sp, q_next)
        
        # =================================================================
        # Step 5. Displace the bulk mesh and update all the meshes. 

        # update the interface mesh
        self.i_mesh.coord_map[:] = self.r

        # we will not update the CL mesh coordinate

        # solve for the bulk mesh displacement
        BMM_basis = FunctionBasis(self.mesh.coord_fe, dx) # for bulk mesh mapping
        A_EL = a_el.assemble(BMM_basis, BMM_basis, dx)
        # impose the displacement of the interface and the sheet
        sol_vec = Function(self.mesh.coord_fe)
        sol_vec[self.BMM_int_dof] = self.r - self.r_m
        sol_vec[self.BMM_sh_dof] = down_to_P1(self.Q_P1_sp, self.q - self.q_m)
        L_EL = Function(self.mesh.coord_fe)   
        L_EL[:] = -A_EL @ sol_vec # homogeneize the boundary conditions

        BMM_free_dof = self.BMM_free_dof
        sol_vec[BMM_free_dof] = spsolve(A_EL[BMM_free_dof][:,BMM_free_dof], L_EL[BMM_free_dof])
        self.mesh.coord_map += sol_vec

        print(Fore.GREEN + "\nt = {:.5f}, ".format((self.step+1) * self.solp.dt) + Style.RESET_ALL, end="")
        print("i-disp = {:.2e}, s-disp = {:.2e}, ".format(
            np.linalg.norm(self.r-self.r_m, ord=np.inf)/dt, np.linalg.norm(self.q-self.q_m, ord=np.inf)/dt), end="")
        print("d_chi = {:+.2e}, slip_cl = ({:+.2e},{:+.2e}), ".format(d_chi/dt, slip_cl[0]/dt, slip_cl[1]/dt), end="")
        print("|m| = {:.4f}, ".format(np.linalg.norm(self.m3)), end="")
        # print("dq={:.4f} ".format(np.linalg.norm(dq)), end="")
        print("dq-={:.4f}, dq+={:.4f}, ".format(np.linalg.norm(dq_minus[self.cl_dof_fd[0]]), np.linalg.norm(dq_plus[self.cl_dof_fd[0]])), end="")

    def finish(self) -> None:
        super().finish()
        if self.args.vis:
            pyplot.ioff()
            pyplot.show()

# ===========================================================

if __name__ == "__main__":
    solp = SolverParameters(dt=1.0/(1024), Te=1.0/4)
    runner = Drop_Runner(solp)
    runner.prepare()
    runner.run()
    runner.finish()
