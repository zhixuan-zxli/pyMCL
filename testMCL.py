from dataclasses import dataclass
import pickle
import numpy as np
from math import cos
from runner import *
from fem import *
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot
from colorama import Fore, Style

@dataclass
class PhysicalParameters:
    eta_2: float = 0.1
    mu_1: float = 0.1
    mu_2: float = 0.1
    mu_cl: float = 0.1
    cosY: float = cos(np.pi*2.0/3)
    gamma_1: float = 5.0
    gamma_3: float = 10.0
    gamma_2: float = 5.0 + 10.0 * cos(np.pi*2.0/3) # to be consistent: gamma_2 = gamma_1 + gamma_3 * cos(theta_Y)
    B: float = 5e-2
    Y: float = 4e2

# ===========================================================
# bilinear forms for the elastic sheet

@Functional
def dx(x: QuadData) -> np.ndarray:
    return x.dx

@Functional
def e_stretch(x: QuadData, w: QuadData) -> np.ndarray:
    strain = w.grad[0,0] + 0.5 * w.grad[1,0]**2 # (Ne, Nq)
    return 0.5 * strain[np.newaxis]**2 * x.dx

@Functional
def e_bend(x: QuadData, mom: QuadData) -> np.ndarray:
    return 0.5 * mom**2 * x.dx

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
    # return (w.grad[0,0] * phi.grad[0,0])[np.newaxis] * x.dx

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

class MCL_Runner(Runner):

    def prepare(self) -> None:
        super().prepare()

        self.phyp = PhysicalParameters()
        with open(self._get_output_name("PhysicalParameters"), "wb") as f:
            pickle.dump(self.phyp, f)
        
        # physical groups from GMSH
        # group_name = {"fluid_1": 1, "fluid_2": 2, "interface": 3, "dry": 4, "wet": 5, \
        #              "right": 6, "top": 7, "left": 8, "cl": 9, "clamp": 10}
        self.mesh = Mesh()
        self.mesh.load("mesh/two-phase.msh")
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
        self.slip_fric = np.where(self.s_mesh.cell_tag[1] == 5, self.phyp.mu_1, self.phyp.mu_2)
        self.surf_tens = np.where(self.s_mesh.cell_tag[1] == 5, self.phyp.gamma_1, self.phyp.gamma_2)

        # =================================================================
        # set up the function spaces
        self.U_sp = FunctionSpace(self.mesh, VectorElement(TriP2, 2))
        self.P1_sp = FunctionSpace(self.mesh, TriP1)
        self.P0_sp = FunctionSpace(self.mesh, TriDG0)
        self.Y_sp = self.i_mesh.coord_fe # type: FunctionSpace # should be VectorElement(LineP1, 2)
        self.K_sp = FunctionSpace(self.i_mesh, LineP1)
        self.Q_sp = FunctionSpace(self.s_mesh, VectorElement(LineP2, 2)) # for deformation and also for the fluid stress
        self.Q_P1_sp = self.s_mesh.coord_fe # type: FunctionSpace
        self.MOM_sp = FunctionSpace(self.s_mesh, LineP2)
        self.M3_sp = FunctionSpace(self.cl_mesh, VectorElement(NodeElement, 2))
        assert self.M3_sp.dof_loc[0,0] < self.M3_sp.dof_loc[2,0]

        # extract the useful DOFs
        self.cl_dof_Q2 = np.unique(self.Q_sp.getFacetDof(tags=(9,)))
        self.cl_dof_Q1 = np.unique(self.Q_P1_sp.getFacetDof(tags=(9,)))
        #
        u_noslip_dof = np.unique(self.U_sp.getFacetDof(tags=(7,)))
        p_fix_dof = None # np.array((0,))
        q_clamp_dof = np.unique(self.Q_sp.getFacetDof(tags=(10,)))
        # q_clamp_dof = np.unique(np.concatenate((Q_sp.getFacetDof(tags=(10,)).reshape(-1), np.arange(1, Q_sp.num_dof, 2)))) # for no-bending
        mom_fix_dof = np.unique(self.MOM_sp.getFacetDof(tags=(10,)))
        # mom_fix_dof = np.arange(MOM_sp.num_dof) # for no-bending
        self.free_dof = group_dof(
            (self.U_sp, self.P1_sp, self.P0_sp, self.Q_sp, self.Y_sp, self.K_sp, self.M3_sp, self.Q_sp, self.MOM_sp), 
            (u_noslip_dof, p_fix_dof, p_fix_dof, None, None, None, None, q_clamp_dof, mom_fix_dof)
        )

        # declare the solution functions
        self.u = Function(self.U_sp)
        self.p1 = Function(self.P1_sp)
        self.p0 = Function(self.P0_sp)
        self.y = Function(self.Y_sp)
        self.y_k = Function(self.Y_sp)
        self.kappa = Function(self.K_sp)
        self.w = Function(self.Q_sp)   # the displacement
        self.w_k = Function(self.Q_sp) # the displacement
        self.tau = Function(self.Q_sp)
        self.mom = Function(self.MOM_sp)
        self.m3 = Function(self.M3_sp)
        self.m3[1::2] = -1.0 # need an initial value for m3
        
        self.id_k = Function(self.s_mesh.coord_fe)
        self.id = Function(self.s_mesh.coord_fe)

        # initialize the arrays for storing the energies
        self.energy = np.zeros((self.num_steps+1, 5))
        # the columns are stretching energy, bending energy, surface energy for Sigma_1, 2, 3. 

        # read checkpoints from file
        if self.args.resume:
            self.mesh.coord_map[:] = self.resume_file["bmm"]
            self.i_mesh.coord_map[:] = self.resume_file["y_k"]
            self.s_mesh.coord_map[:] = self.resume_file["id_k"]
            self.w[:] = self.resume_file["w_k"]
            self.m3[:] = self.resume_file["m3"]
            self.mom[:] = self.resume_file["mom"]
            self.energy[:] = self.resume_file["energy"]
            del self.resume_file

        # prepare visualization
        if self.args.vis:
            pyplot.ion()
            self.ax = pyplot.subplot()
            self.ax.axis("equal")
            self.bulk_triangles = self.mesh.coord_fe.elem_dof[::2,:].T//2
            self.triangle_color = np.where(self.mesh.cell_tag[2] == 1, 1, np.nan)

    def pre_step(self) -> bool:
        # retrieve the mesh mapping; 
        # they will be needed in the measures. 
        self.y_k[:] = self.i_mesh.coord_map
        self.id_k[:] = self.s_mesh.coord_map 
        self.w_k[:] = self.w
        id_k_lift = lift_to_P2(self.Q_sp, self.id_k)
        self.q_k = self.w_k + id_k_lift # type: Function

        # calculate the energy
        dA_k = Measure(self.s_mesh, dim=1, order=5, coord_map=self.id_k) 
        self.energy[self.step, 0] = self.phyp.Y * e_stretch.assemble(dA_k, w=self.w_k._interpolate(dA_k))
        self.energy[self.step, 1] = 1.0/self.phyp.B * e_bend.assemble(dA_k, mom=self.mom._interpolate(dA_k))
        da_1 = Measure(self.s_mesh, dim=1, order=5, tags=(4,), coord_map=self.q_k)
        da_2 = Measure(self.s_mesh, dim=1, order=5, tags=(5,), coord_map=self.q_k)
        self.energy[self.step, 2] = self.phyp.gamma_1 * dx.assemble(da_1)
        self.energy[self.step, 3] = self.phyp.gamma_2 * dx.assemble(da_2)
        ds = Measure(self.i_mesh, dim=1, order=3)
        self.energy[self.step, 4] = self.phyp.gamma_3 * dx.assemble(ds)
        print("  energy={:.5f}, {}".format(np.sum(self.energy[self.step]), self.energy[self.step]))

        t = self.step * self.solp.dt
        if self.args.vis:
            self.ax.clear()
            self.ax.tripcolor(self.mesh.coord_map[::2], self.mesh.coord_map[1::2], self.triangle_color, triangles=self.bulk_triangles)
            self.ax.triplot(self.mesh.coord_map[::2], self.mesh.coord_map[1::2], triangles=self.bulk_triangles, linewidth=0.5)
            # m3_ = m3.view(np.ndarray)
            # ax.quiver(q_k[cl_dof_Q2[::2]], q_k[cl_dof_Q2[1::2]], m3_[::2], m3_[1::2])
            # plot reference sheet mesh
            self.ax.plot(self.id_k[self.cl_dof_Q1[::2]], -0.1*np.ones(2), 'ro')
            self.ax.plot(self.id_k[::2], -0.1*np.ones(self.id_k.size//2), 'b+') 
            self.ax.plot([-1,1], [-0.1,-0.1], 'b-')
            # plot the bending moment
            # self.ax.plot(id_k_lift[::2], mom-0.1, 'kv')
            self.ax.set_ylim(-0.15, 1.0)
            pyplot.title("t={:.5f}".format(t))
            pyplot.draw()
            pyplot.pause(1e-3)
            # output image files
            if self.step % self.solp.stride_frame == 0:
                filename = self._get_output_name("{:04}.png".format(self.step))
                pyplot.savefig(filename, dpi=300.0)
        if self.step % self.solp.stride_checkpoint == 0:
            filename = self._get_output_name("{:04}.npz".format(self.step))
            np.savez(filename, y_k=self.y_k, id_k=self.id_k, w_k=self.w_k, m3=self.m3, \
                     mom = self.mom, bmm=self.mesh.coord_map, energy=self.energy)
            print(Fore.GREEN + "Checkpoint saved to " + filename + Style.RESET_ALL)
        
        return self.step >= self.num_steps
    
    def main_step(self) -> None:
        
        phyp = self.phyp # just for convenience
        # =================================================================
        # Step 1. Update the reference contact line. 

        # extract the current reference CL locations
        chi_k = self.id_k.view(np.ndarray)[self.cl_dof_Q1[::2]] # (2,), with [0] being the left CL
        # ensure [0] is for the left CL; otherwise the code below does not make sense. 
        assert chi_k[0] < chi_k[1]
        
        # project the discontinuous deformation gradient onto P1 to find the conormal vector m1
        dA_k = Measure(self.s_mesh, dim=1, order=5, coord_map=self.id_k) # the reference sheet mesh at the last time step
        q_P1_k_basis = FunctionBasis(self.Q_P1_sp, dA_k)
        C_L2 = c_L2.assemble(q_P1_k_basis, q_P1_k_basis, dA_k)
        L_DQ = l_dq.assemble(q_P1_k_basis, dA_k, q_k = self.q_k._interpolate(dA_k))
        dq_k = Function(self.Q_P1_sp)
        dq_k[:] = spsolve(C_L2, L_DQ)

        # extract the conormal at the contact line
        dq_k_at_cl = dq_k.view(np.ndarray)[self.cl_dof_Q1].reshape(-1, 2) # (-1, 2)
        m1_k = dq_k_at_cl / np.linalg.norm(dq_k_at_cl, ord=None, axis=1, keepdims=True) # (2, 2)
        m1_k[0] = -m1_k[0] 
        # find the displacement of the reference CL driven by unbalanced Young force
        a = np.sum(dq_k_at_cl * m1_k, axis=1) # (2, )
        m3_ = self.m3.view(np.ndarray).reshape(2,2)
        rcl_disp = - solp.dt / (phyp.mu_cl * a) * \
            (phyp.gamma_1-phyp.gamma_2 + phyp.gamma_3 * np.sum(m3_ * m1_k, axis=1))

        # =================================================================
        # Step 2. Find the reference sheet mesh displacement. 

        xx = self.id_k.view(np.ndarray)[::2] # extract the x component of the mesh nodes
        s_mesh_disp = np.where(
            xx <= chi_k[0], (xx + 1.0) / (chi_k[0] + 1.0) * rcl_disp[0], 
            np.where(xx >= chi_k[1], (1.0 - xx) / (1.0 - chi_k[1]) * rcl_disp[1], 
                    (rcl_disp[0] * (chi_k[1] - xx) + rcl_disp[1] * (xx - chi_k[0])) / (chi_k[1] - chi_k[0]))
        )
        eta = Function(self.s_mesh.coord_fe) # the mesh velocity 
        eta[::2] = s_mesh_disp / solp.dt
        id = self.s_mesh.coord_map # type: Function
        id[::2] += s_mesh_disp # update the reference sheet mesh
        id_lift = lift_to_P2(self.Q_sp, id)

        # =================================================================
        # Step 3. Solve the fluid, the fluid-fluid interface, and the sheet deformation. 
        
        # set up the measures
        dx = Measure(self.mesh, dim=2, order=3)
        ds_i = Measure(self.mesh, dim=1, order=3, tags=(3,)) # the fluid interface restricted from the bulk mesh
        ds = Measure(self.i_mesh, dim=1, order=3)
        da_x = Measure(self.mesh, dim=1, order=5, tags=(4, 5)) # the deformed sheet restricted from the bulk mesh

        da_k = Measure(self.s_mesh, dim=1, order=5, coord_map=self.q_k) # the deformed sheet surface measure
        # dA_k = Measure(self.s_mesh, dim=1, order=5, coord_map=id_k) # the reference sheet mesh at the last time step; declared already
        dA = Measure(self.s_mesh, dim=1, order=5) # the reference sheet surface measure at the current time step

        dp_i = Measure(self.i_mesh, dim=0, order=1, tags=(9,)) # the CL restricted from the fluid interfacef
        dp_s = Measure(self.s_mesh, dim=0, order=1, tags=(9,)) # the CL on the reference sheet mesh
        dp_is = Measure(self.s_mesh, dim=0, order=1, tags=(9,), interiorFacet=True)
        dp = Measure(self.cl_mesh, dim=0, order=1)

        # =================================================================
        # set up the function bases:
        # fluid domain
        u_basis = FunctionBasis(self.U_sp, dx)
        p1_basis = FunctionBasis(self.P1_sp, dx)
        p0_basis = FunctionBasis(self.P0_sp, dx)
        # interface domain
        y_basis = FunctionBasis(self.Y_sp, ds)
        u_i_basis = FunctionBasis(self.U_sp, ds_i)
        k_basis = FunctionBasis(self.K_sp, ds)
        # boundary of interface (CL)
        y_cl_basis = FunctionBasis(self.Y_sp, dp_i)
        # sheet
        u_b_basis = FunctionBasis(self.U_sp, da_x)
        q_surf_basis = FunctionBasis(self.Q_sp, da_k)
        q_basis = FunctionBasis(self.Q_sp, dA)
        mom_basis = FunctionBasis(self.MOM_sp, dA)
        q_k_basis = FunctionBasis(self.Q_sp, dA_k) # also the basis for tau
        # CL from sheet
        q_cl_basis = FunctionBasis(self.Q_sp, dp_s)
        q_icl_basis = FunctionBasis(self.Q_sp, dp_is)
        mom_cl_basis = FunctionBasis(self.MOM_sp, dp_is)
        # CL
        m3_basis = FunctionBasis(self.M3_sp, dp)
        
        # =================================================================
        # assemble by blocks
        A_XIU = a_xiu.assemble(u_basis, u_basis, dx, eta=self.viscosity)
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
                                q_k = self.q_k._interpolate(dA_k), mu_i=self.slip_fric)
        L_PI_ADV = l_pi_adv.assemble(q_k_basis, dA_k, q_k = self.q_k._interpolate(dA_k), eta_k = eta._interpolate(dA_k))
        L_PI_Q = l_pi_L2.assemble(q_k_basis, dA_k, q_k=(self.q_k-id_lift)._interpolate(dA_k))

        #C_PHITAU = B_PIQ.T
        #C_PHIM3 = A_M3Q.T
        C_CL_PHIM = c_cl_phim.assemble(q_icl_basis, mom_cl_basis, dp_is)
        C_PHIM = c_phim.assemble(q_basis, mom_basis, dA)
        C_PHIQ = c_phiq.assemble(q_basis, q_basis, dA, w_k = self.w_k._interpolate(dA))
        C_PHIQ_SURF = c_phiq_surf.assemble(q_surf_basis, q_surf_basis, da_k, gamma=self.surf_tens)
        # C_CL_WQ = C_CL_PHIM.T
        # C_WQ = C_PHIM.T
        C_WM = c_L2.assemble(mom_basis, mom_basis, dA)

        # collect the block matrices
        #    u,        p1,      p0,      tau,      y,       k,      m3,        w,      m
        A = bmat((
            (A_XIU,    -A_XIP1, -A_XIP0, A_XITAU,  None,    -phyp.gamma_3*A_XIK, None, None,   None),  # u
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
        self.u[:] = 0.0; self.p1[:] = 0.0; self.p0[:] = 0.0
        self.tau[:] = solp.dt * L_PI_ADV + L_PI_Q
        self.y[:] = 0.0; self.kappa[:] = A_ZK.T @ self.y_k
        self.m3[:] = L_M3
        self.w[:] = C_PHIQ_SURF @ id_lift
        self.mom[:] = 0.0
        L = group_fn(self.u, self.p1, self.p0, self.tau, self.y, \
                     self.kappa, self.m3, self.w, self.mom)

        # the essential boundary conditions are all homogeneous, 
        # so no need to homogeneize the right-hand-side.

        # solve the coupled system
        free_dof = self.free_dof
        sol_free = spsolve(A[free_dof][:,free_dof], L[free_dof])
        sol_full = np.zeros_like(L)
        sol_full[free_dof] = sol_free
        split_fn(sol_full, \
                 self.u, self.p1, self.p0, self.tau, self.y, \
                 self.kappa, self.m3, self.w, self.mom)
        q = self.w + id_lift
        
        # force unit length of m3
        # m3_ = m3.view(np.ndarray).reshape(2,2)
        # m3_ = m3_ / np.linalg.norm(m3_, axis=1, keepdims=True)
        # m3[:] = m3_.reshape(-1)
        
        # =================================================================
        # Step 4. Displace the bulk mesh and update all the meshes. 

        # update the interface mesh
        self.i_mesh.coord_map[:] = self.y

        # no need to update the CL mesh as the coordinates are never used

        # the sheet mesh is already update in Step 2

        # solve for the bulk mesh displacement
        BMM_basis = FunctionBasis(self.mesh.coord_fe, dx) # for bulk mesh mapping
        A_EL = a_el.assemble(BMM_basis, BMM_basis, dx)
        # attach the displacement of the interface and the sheet
        BMM_int_dof = np.unique(self.mesh.coord_fe.getFacetDof(tags=(3,)))
        BMM_s_dof = np.unique(self.mesh.coord_fe.getFacetDof(tags=(4,5)))
        BMM_fix_dof = np.concatenate(self.mesh.coord_fe.getFacetDof(tags=(3,4,5,6,7,8)))
        bulk_disp = Function(self.mesh.coord_fe)
        bulk_disp[BMM_int_dof] = self.y - self.y_k
        bulk_disp[BMM_s_dof] = down_to_P1(self.Q_P1_sp, q - self.q_k)
        L_EL = Function(self.mesh.coord_fe)
        L_EL[:] = -A_EL @ bulk_disp # homogeneize the boundary conditions

        el_free_dof = group_dof((self.mesh.coord_fe,), (BMM_fix_dof,))
        bulk_disp[el_free_dof] = spsolve(A_EL[el_free_dof][:,el_free_dof], L_EL[el_free_dof])
        self.mesh.coord_map += bulk_disp

        print(Fore.GREEN + "t = {:.5f}, ".format((self.step+1) * self.solp.dt) + Style.RESET_ALL, end="")
        print("i-disp = {:.2e}, s-disp = {:.2e}, ".format(
            np.linalg.norm(self.y-self.y_k, ord=np.inf), np.linalg.norm(q-self.q_k, ord=np.inf)), end="")
        print("cl = ({:.4f}, {:.4f}), ".format(q[self.cl_dof_Q2[0]], q[self.cl_dof_Q2[2]]), end="")
        print("|m| = ({:.4f}, {:.4f})".format(np.sqrt(self.m3[0]**2+self.m3[1]**2), np.sqrt(self.m3[2]**2+self.m3[3]**2)))

    def finish(self) -> None:
        super().finish()
        if self.args.vis:
            pyplot.ioff()
            pyplot.show()

# ===========================================================

if __name__ == "__main__":
    solp = SolverParameters(dt=1.0/1024/16, Te=0.5)
    MCL_Runner(solp).run()
