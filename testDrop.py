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
    mu_1: float = 1e3
    mu_2: float = 1e3
    mu_cl: float = 1.0
    gamma_1: float = 0.0
    gamma_3: float = 5.0
    gamma_2: float = 0.0 + 5.0 * cos(np.pi/3) # to be consistent: gamma_2 = gamma_1 + gamma_3 * cos(theta_Y)
    Cb: float = 1e-2
    Cs: float = 1e2
    pre: float = 0.1 # the initial Jacobian is 1 + pre

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

# For a_etaq, use a_pir
# For a_etak, use a_L2

# For a_xikappa, use a_pir

@LinearForm
def l_xi(xi: QuadData, x: QuadData, gamma: np.ndarray) -> np.ndarray:
    # gamma: (Nf, )
    return (gamma[:,np.newaxis] * (xi.grad[0,0] + xi.grad[1,1]))[np.newaxis] * x.dx

@LinearForm
def l_wm_xi(xi: QuadData, x: QuadData, k_m: QuadData) -> np.ndarray:
    # k_m: (2, Nf, Nq)
    # xi.grad: (2, 2, Nf, Nq)
    return (k_m[0]**2 + k_m[1]**2) * (xi.grad[0,0] + xi.grad[1,1])[np.newaxis] * x.dx

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
def a_slip(v: QuadData, u: QuadData, _, x: QuadData, mu: np.ndarray, x2: QuadData) -> np.ndarray:
    # v, u: (2, Nf, Nq)
    # x2.cn: (2, Nf, Nq)
    tau = np.array((-x2.cn[1], x2.cn[0]))
    v_t = np.sum(v * tau, axis=0)
    u_t = np.sum(u * tau, axis=0) # (Nf, Nq)
    return (mu[:, np.newaxis]*v_t*u_t)[np.newaxis] * x2.dx

@BilinearForm
def a_vt(v: QuadData, t: QuadData, x: QuadData, _, x2: QuadData) -> np.ndarray:
    return np.sum(v * x2.cn, axis=0, keepdims=True) * t * x2.dx

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
        self.T_sp = FunctionSpace(self.s_mesh, LineP1) # for the normal stress
        self.M3_sp = FunctionSpace(self.cl_mesh, VectorElement(NodeElement, 2))

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
        q_clamp_dof = np.where(self.Q_sp.dof_loc[:,0] > 1-1e-12)[0]
        q_sym_dof = np.where(self.Q_sp.dof_loc[:,0] < 1e-12)[0]
        # q_all_dof = np.arange(self.Q_sp.num_dof)
        q_fix_dof = np.unique(np.concatenate((q_clamp_dof, q_sym_dof[0:1])))
        q_P1_clamp_dof = np.where(self.Q_P1_sp.dof_loc[:,0] > 1-1e-12)[0]
        q_P1_sym_dof = np.where(self.Q_P1_sp.dof_loc[:,0] < 1e-12)[0]
        q_P1_fix_dof = np.unique(np.concatenate((q_P1_clamp_dof, q_P1_sym_dof[0:1])))
        self.k_m_free_dof = group_dof((self.Q_P1_sp,), (q_P1_fix_dof,))
        self.free_dof = group_dof(
            (self.U_sp, self.P1_sp, self.P0_sp, self.R_sp, self.OMG_sp, self.Q_sp, self.Q_P1_sp, self.T_sp, self.M3_sp), 
            (u_fix_dof, None, p_fix_dof, r_sym_dof[0], None, q_fix_dof, q_P1_fix_dof, None, None) 
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
        self.dqdt = Function(self.Q_sp) # the velocity of the deformation map, for plotting only
        self.kappa = Function(self.Q_P1_sp) # the mean curvature vector
        self.k_m = Function(self.Q_P1_sp)
        self.tnn = Function(self.T_sp) # the normal stress
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
        self.energy = np.zeros((self.num_steps+1, 5)) # the columns are stretching energy, bending energy, surface energy for Sigma_1, 2, 3. 
        self.phycl_hist = np.zeros((self.num_steps+1, 2)) # history of physical CL
        self.refcl_hist = np.zeros((self.num_steps+1, 2)) # history of reference CL
        self.thd_hist = np.zeros((self.num_steps+1, ))    # history of the dynamic contact angle

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
            self.thd_hist[:self.step+1] = self.resume_file["thd_hist"]
            del self.resume_file

        # prepare visualization
        if self.args.vis:
            pyplot.ion()
            pyplot.rc("font", size=16)
            self.fig, self.ax = pyplot.subplots()
            self.fig.set_size_inches(6, 7.2)
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
        self.thd_hist[self.step] = np.dot(self.m1_hat, self.m3)

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
            self.ax.quiver(self.mesh.coord_map[::2], self.mesh.coord_map[1::2], _u[:_n:2], _u[1:_n:2], color="tab:blue") #, scale=5.0, scale_units='x')
            # plot the interface
            _r_m = self.r_m.view(np.ndarray)
            segments = _r_m[self.i_mesh.coord_fe.elem_dof].reshape(2, 2, -1).transpose(2, 0, 1)
            self.ax.add_collection(LineCollection(segments=segments, colors="tab:green"))
            # plot the sheet
            _q_m_down = self.q_m_down.view(np.ndarray)
            segments = _q_m_down[self.s_mesh.coord_fe.elem_dof].reshape(2, 2, -1).transpose(2, 0, 1)
            self.ax.add_collection(LineCollection(segments=segments, colors="tab:orange"))
            # self.ax.plot(self.q_m[::2], self.q_m[1::2], "k+")
            # self.ax.plot(self.q_m[::2], self.dqdt[::2], 'ro', mfc='none')
            # self.ax.plot(self.q_m[::2], self.dqdt[1::2], 'bo', mfc='none')
            # plot the cornormals
            # self.ax.quiver(phycl[0], phycl[1], self.m1_hat[0], self.m1_hat[1], color="tab:brown") # m1
            # self.ax.quiver(phycl[0], phycl[1], self.m3[0], self.m3[1], color="tab:brown") # m3
            # plot the frame
            # self.ax.plot((0.0, 1.0, 1.0), (1.0, 1.0, 0.0), 'k-')
            self.ax.set_xlim(0.0, 1.0); self.ax.set_ylim(-0.2, 1.0)
            # pyplot.title("t={:.5f}".format(t))
            pyplot.draw()
            pyplot.pause(1e-4)
            # output image files
            if self.step % self.solp.stride_frame == 0:
                filename = self._get_output_name("{:05}.png".format(self.step))
                self.fig.savefig(filename, dpi=300.0, bbox_inches="tight")
        if self.step % self.solp.stride_checkpoint == 0:
            filename = self._get_output_name("{:05}.npz".format(self.step))
            np.savez(filename, bulk_coord_map=self.mesh.coord_map, r_m=self.r_m, id_m=self.id_m, q_m=self.q_m, k_m=self.kappa, 
                     phycl_hist=self.phycl_hist[:self.step+1], 
                     refcl_hist=self.refcl_hist[:self.step+1], 
                     thd_hist = self.thd_hist[:self.step+1],
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
        tnn_basis = FunctionBasis(self.T_sp, da)
        q_da_basis = FunctionBasis(self.Q_sp, da)
        q_P1_da_basis = FunctionBasis(self.Q_P1_sp, da)
        q_ref_basis = FunctionBasis(self.Q_sp, d_xi)
        # CL from sheet
        r_cl_basis = FunctionBasis(self.R_sp, dp_i)
        q_cl_basis = FunctionBasis(self.Q_sp, dp_s)
        m3_basis = FunctionBasis(self.M3_sp, dp)
        
        # assemble by blocks
        # for the Stokes equation
        A_VU = a_vu.assemble(u_basis, u_basis, None, None, eta=self.viscosity)
        A_VP1 = a_vp.assemble(u_basis, p1_basis)
        A_VP0 = a_vp.assemble(u_basis, p0_basis)
        A_VU_S = a_slip.assemble(u_da_basis, u_da_basis, None, None, mu=self.slip_fric, x2=da.x)
        A_VQ = a_slip.assemble(u_da_basis, q_da_basis, None, None, mu=self.slip_fric, x2=da.x)
        A_VT = a_vt.assemble(u_da_basis, tnn_basis, None, None, x2=da.x)
        # for the fluid interface
        A_VOMG = a_vomg.assemble(u_ds_basis, omega_basis)
        A_PIR = a_pir.assemble(r_basis, r_basis)
        A_PIOMG = a_piomg.assemble(r_basis, omega_basis)
        A_PIM3 = a_pim3.assemble(r_cl_basis, m3_basis)
        # for the compatibility of the sheet
        A_ETAQ = a_pir.assemble(q_P1_da_basis, q_da_basis)
        A_ETAK = a_L2.assemble(q_P1_da_basis, q_P1_da_basis)
        # for the mechanics of the sheet
        A_XIT = a_vt.assemble(q_da_basis, tnn_basis, None, None, x2=da.x)
        A_XIQ_S = a_slip.assemble(q_da_basis, q_da_basis, None, None, mu=self.slip_fric, x2=da.x)
        A_XIK = a_pir.assemble(q_da_basis, q_P1_da_basis)
        A_XIQ = a_xiq.assemble(q_ref_basis, q_ref_basis, None, None, q_m=self.q_m._interpolate(d_xi))
        A_XIQ_2 = a_xiq_2.assemble(q_ref_basis, q_ref_basis, None, None, q_m=self.q_m._interpolate(d_xi))
        L_XI = l_xi.assemble(q_da_basis, None, gamma=self.surf_tens)
        A_XIM3 = a_pim3.assemble(q_cl_basis, m3_basis)
        # for the contact line condition
        A_M3M3 = a_m3m3.assemble(m3_basis, m3_basis, None, None, m1_hat=self.m1_hat._interpolate(dp))
        L_M3 = l_m3.assemble(m3_basis, None, m1_hat=self.m1_hat._interpolate(dp))

        # solve for the sheet curvature explicitly
        k_m_free_dof = self.k_m_free_dof
        self.k_m[:] = 0.0
        sol_free = spsolve(A_ETAK[k_m_free_dof][:,k_m_free_dof], -(A_ETAQ @ self.q_m)[k_m_free_dof])
        self.k_m[k_m_free_dof] = sol_free
        L_WM_XI = l_wm_xi.assemble(q_da_basis, None, k_m=self.k_m._interpolate(da))

        # collect the block matrices
        dt = solp.dt
        #    u,             p1,     p0,       r,      omega,                q,       kappa,   tnn,    m3
        A = bmat((
            (A_VU+A_VU_S,   -A_VP1, -A_VP0,   None,   -phyp.gamma_3*A_VOMG, -A_VQ,   None,    -A_VT,  None),     # v
            (A_VP1.T,       None,   None,     None,   None,                 None,    None,    None,   None),     # rho_1
            (A_VP0.T,       None,   None,     None,   None,                 None,    None,    None,   None),     # rho_0
            (None,          None,   None,     A_PIR,  A_PIOMG,              None,    None,    None,   -A_PIM3),  # pi
            (-dt*A_VOMG.T,  None,   None,     A_PIOMG.T,  None,             None,    None,    None,   None),     # delta
            (-A_VQ.T,       None,   None,     None,   None,   phyp.Cs*dt*(A_XIQ+A_XIQ_2)+A_XIQ_S, -phyp.Cb*A_XIK, A_XIT, phyp.gamma_3*A_XIM3), # xi
            (None,          None,   None,     None,   None,                 dt*A_ETAQ, A_ETAK, None,  None),     # eta
            (-A_VT.T,       None,   None,     None,   None,                 A_XIT.T, None,    None,   None),     # chi
            (None,          None,   None,     phyp.mu_cl/dt*A_PIM3.T, None, -phyp.mu_cl*A_XIM3.T, None, None, phyp.gamma_3*A_M3M3),    # m3
        ), format="csc")
        # collect the right-hand side
        self.u[:] = 0.0
        self.p1[:] = 0.0
        self.p0[:] = 0.0
        self.r[:] = 0.0
        self.omega[:] = A_PIOMG.T @ self.r_m
        self.q[:] = -L_XI - phyp.Cs * (A_XIQ @ self.q_m) + 1.5 * phyp.Cb * L_WM_XI
        self.kappa[:] = -(A_ETAQ @ self.q_m)
        self.tnn[:] = 0.0
        self.m3[:] = (phyp.gamma_2 - phyp.gamma_1) * L_M3 + phyp.mu_cl/dt*A_PIM3.T @ self.r_m
        L = group_fn(self.u, self.p1, self.p0, self.r, self.omega, self.q, self.kappa, self.tnn, self.m3)
        # set up the boundary conditions and homogeneize the right-hand side; do nothing since the boundary conditions are homogeneneous
        sol_full = np.zeros_like(L)
        # solve the coupled system
        free_dof = self.free_dof
        sol_free = spsolve(A[free_dof][:,free_dof], L[free_dof])
        sol_full[free_dof] = sol_free
        split_fn(sol_full, self.u, self.p1, self.p0, self.r, self.omega, self.q, self.kappa, self.tnn, self.m3)

        # q is the deformation velocity, and update to get the deformation map
        np.copyto(self.dqdt, self.q)
        self.q[:] = self.q_m + dt * self.q
        
        # =================================================================
        # Step 2. Convert to FD form and find the derivatives using finite difference 
        q_fd = arrange_as_FD(self.Q_sp, self.q) # (n, 2)
        xi_fd = arrange_as_FD(self.Q_sp, lift_to_P2(self.Q_sp, self.s_mesh.coord_map))[:,0][:, np.newaxis] # (n, 1)
        assert np.all(xi_fd[1:,0] - xi_fd[:-1,0] > 0), "The mesh mapping is not strictly increasing!"
        assert np.all(q_fd[1:,0] - q_fd[:-1,0] > 0), "The mesh mapping is not strictly increasing!"

        slip_cl = self.r.view(np.ndarray)[self.cl_dof_R] - self.q.view(np.ndarray)[self.cl_dof_Q] # type: np.ndarray # (2,)
        j = self.cl_dof_fd[0]
        # find the reference CL velocity using an upwind derivative
        h_r = xi_fd[j+1] - xi_fd[j]
        dq_plus = (q_fd[j+1] - q_fd[j]) / h_r * (3/2) - (q_fd[j+2] - q_fd[j+1]) / h_r * (1/2)
        h_l = xi_fd[j] - xi_fd[j-1]
        dq_minus = -(q_fd[j-1] - q_fd[j-2]) / h_l * (1/2) + (q_fd[j] - q_fd[j-1]) / h_l * (3/2)
        if slip_cl[0] > 0:
            d_chi = np.dot(slip_cl, dq_plus) / np.sum(dq_plus**2)
        else:
            d_chi = np.dot(slip_cl, dq_minus) / np.sum(dq_minus**2)
        # adjust the reference mesh mapping
        ref_cl = xi_fd[j] # (1,)
        d_xi = np.where(
            xi_fd <= ref_cl[0], xi_fd / ref_cl[0] * d_chi, (xi_fd[-1]-xi_fd) / (xi_fd[-1]-ref_cl[0]) * d_chi
        ) # (n, )

        def _reinterpolate(q_fd: np.ndarray) -> np.ndarray:
            q_adv = q_fd.copy()
            # interpolate the middle nodes
            h = (xi_fd[2::2] - xi_fd[:-2:2])/2
            x = d_xi[1:-1:2]
            q_adv[1:-1:2] = (q_fd[:-2:2] * (x*(x-h)/2) - q_fd[1:-1:2] * ((x+h)*(x-h)) + q_fd[2::2] * ((x+h)*x/2)) / h**2
            # interpolate to the right
            hr = h[1:]
            x = d_xi[2:-2:2]-hr
            q_r = (q_fd[2:-2:2] * (x*(x-hr)/2) - q_fd[3:-1:2] * ((x+hr)*(x-hr)) + q_fd[4::2] * ((x+hr)*x/2)) / hr**2
            # interpolate to the left
            hl = h[:-1]
            x = d_xi[2:-2:2]+hl
            q_l = (q_fd[:-4:2] * (x*(x-hl)/2) - q_fd[1:-3:2] * ((x+hl)*(x-hl)) + q_fd[2:-2:2] * ((x+hl)*x/2)) / hl**2
            q_adv[2:-2:2] = np.where(d_xi[2:-2:2] > 0, q_r, q_l)
            return q_adv
        q_adv = _reinterpolate(q_fd)
        q_adv[j] = self.r.view(np.ndarray)[self.cl_dof_R] 
        self.q[:] = arrange_as_FE(self.Q_sp, q_adv)
        
        # now overwrite the mesh mapping
        self.s_mesh.coord_map += down_to_P1(self.Q_P1_sp, arrange_as_FE(self.Q_sp, np.concatenate((d_xi, np.zeros_like(d_xi)), axis=1)))
        
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
        print("i-vel = {:.2e}, s-vel = {:.2e}, ".format(
            np.linalg.norm(self.r-self.r_m, ord=np.inf)/dt, np.linalg.norm(self.q-self.q_m, ord=np.inf)/dt), end="")
        print("ref-cl-vel = {:+.2e}, phyp-cl-vel = ({:+.2e},{:+.2e}), ".format(d_chi/dt, slip_cl[0]/dt, slip_cl[1]/dt), end="")
        print("|m| = {:.4f}, ".format(np.linalg.norm(self.m3)), end="")
        print("dq-={:.4f}, dq+={:.4f}, ".format(np.linalg.norm(dq_minus), np.linalg.norm(dq_plus)), end="")

    def finish(self) -> None:
        super().finish()
        if self.args.vis:
            pyplot.ioff()
            pyplot.show()

# ===========================================================

if __name__ == "__main__":
    solp = SolverParameters(dt=1.0/(8192), Te=4.0)
    runner = Drop_Runner(solp)
    runner.prepare()
    runner.run()
    runner.finish()
