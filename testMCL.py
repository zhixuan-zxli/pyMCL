import pickle
import numpy as np
from math import cos
from runner import *
from fem import *
from scipy.sparse import bmat
from scikits.umfpack import spsolve
from matplotlib import pyplot
from colorama import Fore, Style

from testDrop import PhysicalParameters, \
    dx, e_stretch, e_bend, \
    c_L2, l_dq, c_phim, c_phiq, c_phiq_surf, b_pitau, \
    a_xiu, a_xip, a_xik, a_zy, a_zk, a_zm3, a_m3m3, a_el, \
    lift_to_P2, down_to_P1, arrange_as_FD, arrange_as_FE

@dataclass
class PhysicalParameters_MCL(PhysicalParameters):
    U_adv: float = 0.1

@BilinearForm
def b_pidq(pi: QuadData, q: QuadData, x: QuadData) -> np.ndarray:
    # pi: (2, Ne, Nq)
    # q.grad(2, 2, Ne, Nq)
    return np.sum(q.grad[:,0] * pi, axis=0, keepdims=True) * x.dx

@LinearForm
def l_m3_adv(m3: QuadData, x: QuadData, dq_k: QuadData) -> np.ndarray:
    # m3: (2, Ne, Nq)
    # dq_k: (2, Ne, Nq)
    return np.sum(m3 * dq_k, axis=0, keepdims=True) * x.dx

# ===========================================================

class MCL_Runner(Runner):

    def prepare(self) -> None:
        super().prepare()

        self.phyp = PhysicalParameters_MCL(
            eta_2=0.1, mu_1=0.1, mu_2=0.1, mu_cl=0.1, 
            gamma_1=2.5, gamma_3=5.0, gamma_2 = 2.5 + 5.0 * cos(np.pi/2), 
            B=1e-2, Y=1e2, U_adv=0.0
        )
        with open(self._get_output_name("PhysicalParameters"), "wb") as f:
            pickle.dump(self.phyp, f)
        
        # physical groups from GMSH
        # group_name = {
        #     "fluid_1": 1, "fluid_2": 2, "interface": 3, 
        #     "wet_sheet": 4, "dry_sheet": 5, "wet_top": 6, "drop_top": 7, 
        #     "left": 8, "right": 9, "cl": 10, "cl_top": 11, "clamp": 12
        # }
        self.mesh = Mesh()
        self.mesh.load(self.args.mesh_name)
        for _ in range(self.args.spaceref):
            self.mesh = splitRefine(self.mesh)
        setMeshMapping(self.mesh)
        self.i_mesh = self.mesh.view(1, tags=(3, )) # interface mesh
        setMeshMapping(self.i_mesh)
        self.s_mesh = self.mesh.view(1, tags=(4, 5)) # sheet reference mesh
        setMeshMapping(self.s_mesh)
        self.cl_mesh = self.mesh.view(0, tags=(10, )) # contact line mesh
        setMeshMapping(self.cl_mesh)

        # prepare the variable coefficients
        self.viscosity = np.where(self.mesh.cell_tag[2] == 1, 1.0, self.phyp.eta_2)
        self.slip_fric = np.where(self.s_mesh.cell_tag[1] == 4, self.phyp.mu_1, self.phyp.mu_2)
        self.surf_tens = np.where(self.s_mesh.cell_tag[1] == 4, self.phyp.gamma_1, self.phyp.gamma_2)

        # =================================================================
        # set up the function spaces
        self.U_sp = FunctionSpace(self.mesh, VectorElement(TriP2, 2))
        self.P1_sp = FunctionSpace(self.mesh, TriP1)
        self.P0_sp = FunctionSpace(self.mesh, TriDG0)
        self.Y_sp = self.i_mesh.coord_fe # type: FunctionSpace # should be VectorElement(LineP1, 2)
        self.K_sp = FunctionSpace(self.i_mesh, LineP1)
        self.Q_sp = FunctionSpace(self.s_mesh, VectorElement(LineP2, 2)) # for sheet deformation
        self.Q_P1_sp = self.s_mesh.coord_fe # type: FunctionSpace
        self.TAU_sp = FunctionSpace(self.s_mesh, VectorElement(LineDG0, 2)) # for the stress
        self.MOM_sp = FunctionSpace(self.s_mesh, LineP2)
        self.M3_sp = FunctionSpace(self.cl_mesh, VectorElement(NodeElement, 2))

        # extract the useful DOFs
        self.cl_top_dof_K = np.unique(self.K_sp.getFacetDof(tags=(11,)))
        self.cl_top_dof_Y = np.unique(self.Y_sp.getFacetDof(tags=(11,)))
        self.cl_dof_Y = np.unique(self.Y_sp.getFacetDof(tags=(10,)))
        self.cl_dof_Q2 = np.unique(self.Q_sp.getFacetDof(tags=(10,)))
        self.cl_dof_Q1 = np.unique(self.Q_P1_sp.getFacetDof(tags=(10,)))
        #
        u_noslip_dof = np.unique(self.U_sp.getFacetDof(tags=(6, 7)))
        p_fix_dof = np.array((0,), dtype=np.int32) # np.arange(self.P0_sp.num_dof, dtype=np.int32)
        q_clamp_dof = np.unique(self.Q_sp.getFacetDof(tags=(12,)))
        # q_clamp_dof = np.unique(np.concatenate((self.Q_sp.getFacetDof(tags=(10,)).reshape(-1), np.arange(1, self.Q_sp.num_dof, 2)))) # for no-bending
        mom_fix_dof = np.unique(self.MOM_sp.getFacetDof(tags=(12,)))
        # mom_fix_dof = np.arange(self.MOM_sp.num_dof) # for no-bending
        self.free_dof = group_dof(
            (self.U_sp, self.P1_sp, self.P0_sp, self.TAU_sp, self.Y_sp, self.K_sp, self.M3_sp, self.Q_sp, self.MOM_sp), 
            (u_noslip_dof, None, p_fix_dof, None, self.cl_top_dof_Y, self.cl_top_dof_K, None, q_clamp_dof, mom_fix_dof)
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
        self.tau = Function(self.TAU_sp)
        self.mom = Function(self.MOM_sp)
        self.m3 = Function(self.M3_sp)
        self.m1_k = Function(self.M3_sp) # the projected m1
        self.id_k = Function(self.s_mesh.coord_fe)

        # mark the CL node for finite difference arrangement
        _temp = -np.ones_like(self.w, dtype=int)
        _temp[self.cl_dof_Q2] = (0, 1)
        self.cl_dof_fd = arrange_as_FD(self.Q_sp, _temp)

        # initialize the arrays for storing the energies
        self.energy = np.zeros((self.num_steps+1, 5))
        # ^ the columns are stretching energy, bending energy, surface energy for Sigma_1, 2, 3. 
        self.phycl_hist = np.zeros((self.num_steps+1, 2)) # history of physical CL
        self.refcl_hist = np.zeros((self.num_steps+1, 2)) # history of reference CL

        # read checkpoints from file
        if self.args.resume:
            self.mesh.coord_map[:] = self.resume_file["bulk_coord_map"]
            self.i_mesh.coord_map[:] = self.resume_file["y_k"]
            self.s_mesh.coord_map[:] = self.resume_file["id_k"]
            self.w[:] = self.resume_file["w_k"]
            self.m3[:] = self.resume_file["m3"]
            self.m1_k[:] = self.resume_file["m1"]
            self.mom[:] = self.resume_file["mom"]
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
        # retrieve the mesh mapping; 
        # they will be needed in the measures. 
        self.y_k[:] = self.i_mesh.coord_map
        self.id_k[:] = self.s_mesh.coord_map 
        self.w_k[:] = self.w
        self.id_k_lift = lift_to_P2(self.Q_sp, self.id_k) # type: Function
        self.q_k = self.w_k + self.id_k_lift # type: Function
        self.q_k_down = down_to_P1(self.Q_P1_sp, self.q_k) # type: Function

        # record the CL locations
        refcl = self.id_k.view(np.ndarray)[self.cl_dof_Q1]
        phycl = self.w_k.view(np.ndarray)[self.cl_dof_Q2]
        phycl += refcl
        self.phycl_hist[self.step] = phycl
        self.refcl_hist[self.step] = refcl

        # calculate the energy
        dA = Measure(self.s_mesh, dim=1, order=5, coord_map=self.id_k) 
        self.energy[self.step, 0] = self.phyp.Y * e_stretch.assemble(dA, w=self.w_k._interpolate(dA))
        self.energy[self.step, 1] = 1.0/self.phyp.B * e_bend.assemble(dA, mom=self.mom._interpolate(dA))
        da_1 = Measure(self.s_mesh, dim=1, order=5, tags=(4,), coord_map=self.q_k)
        self.energy[self.step, 2] = self.phyp.gamma_1 * dx.assemble(da_1)
        da_2 = Measure(self.s_mesh, dim=1, order=5, tags=(5,), coord_map=self.q_k)
        self.energy[self.step, 3] = self.phyp.gamma_2 * dx.assemble(da_2)
        ds = Measure(self.i_mesh, dim=1, order=3)
        self.energy[self.step, 4] = self.phyp.gamma_3 * dx.assemble(ds)
        print("energy={:.5f}, ".format(np.sum(self.energy[self.step])), end="")

        t = self.step * self.solp.dt
        if self.args.vis:
            self.ax.clear()
            # self.ax.tripcolor(self.mesh.coord_map[::2], self.mesh.coord_map[1::2], self.triangle_color, triangles=self.bulk_triangles)
            press = self.p0.view(np.ndarray)[self.P0_sp.elem_dof][0] + np.sum(self.p1.view(np.ndarray)[self.P1_sp.elem_dof], axis=0) / 3 # (Nt, )
            tpc = self.ax.tripcolor(self.mesh.coord_map[::2], self.mesh.coord_map[1::2], press, triangles=self.bulk_triangles)
            if not hasattr(self, "colorbar"):
                self.colorbar = pyplot.colorbar(tpc)
            else:
                self.colorbar.update_normal(tpc) # update the scale of the color bar without redrawing it
            self.ax.triplot(self.mesh.coord_map[::2], self.mesh.coord_map[1::2], triangles=self.bulk_triangles, linewidth=0.5)
            # plot the velocity
            # _u = self.u.view(np.ndarray); _n = self.mesh.coord_map.size
            # self.ax.quiver(self.mesh.coord_map[::2], self.mesh.coord_map[1::2], _u[:_n:2], _u[1:_n:2])
            # plot the conormal
            m3_ = self.m3.view(np.ndarray)
            self.ax.quiver(self.q_k[self.cl_dof_Q2[::2]], self.q_k[self.cl_dof_Q2[1::2]], m3_[::2], m3_[1::2], color="tab:pink")
            # plot reference sheet mesh
            self.ax.plot(self.id_k[self.cl_dof_Q1[::2]], -0.1*np.ones(1), 'ro')
            self.ax.plot(self.id_k[::2], self.id_k[1::2] - 0.1, 'b+') 
            self.ax.plot([-1,1], [-0.1,-0.1], 'b-')
            # plot the fluid stress
            # _x = self.id_k.view(np.ndarray)[self.Q_P1_sp.elem_dof[0::2]].T # (Ne, 2)
            # _x = np.sum(_x, axis=1)/2 # (Ne, )
            # _y0 = self.tau.view(np.ndarray)[self.TAU_sp.elem_dof[0]] # (Ne, )
            # _y0_max = max(np.linalg.norm(_y0, ord=np.inf), 1)
            # _y1 = self.tau.view(np.ndarray)[self.TAU_sp.elem_dof[1]]
            # _y1_max = max(np.linalg.norm(_y1, ord=np.inf), 1)
            # self.ax.plot(_x, _y0/_y0_max, marker="_", color="tab:orange")
            # self.ax.plot(_x, _y1/_y1_max, marker="|", color="tab:pink")
            # self.ax.add_collection(LineCollection(segments=np.dstack((_x, _y0/_y0_max)), colors="tab:orange"))
            # self.ax.add_collection(LineCollection(segments=np.dstack((_x, _y1/_y1_max)), colors="tab:pink"))
            # plot the bending moment
            # self.ax.plot(self.id_k_lift[::2], self.mom-0.1, 'kv')
            self.ax.set_ylim(-0.15, 1.0)
            pyplot.title("t={:.5f}".format(t))
            pyplot.draw()
            pyplot.pause(1e-4)
            # output image files
            if self.step % self.solp.stride_frame == 0:
                filename = self._get_output_name("{:04}.png".format(self.step))
                pyplot.savefig(filename, dpi=300.0)
        if self.step % self.solp.stride_checkpoint == 0:
            filename = self._get_output_name("{:04}.npz".format(self.step))
            np.savez(filename, bulk_coord_map=self.mesh.coord_map, y_k=self.y_k, id_k=self.id_k, \
                     w_k=self.w_k, m3=self.m3, m1=self.m1_k, mom=self.mom,  \
                     phycl_hist=self.phycl_hist[:self.step+1], \
                     refcl_hist=self.refcl_hist[:self.step+1], \
                     energy=self.energy[:self.step+1])
            print(Fore.GREEN + "\nCheckpoint saved to " + filename + Style.RESET_ALL)
        
        return self.step >= self.num_steps
    
    def main_step(self) -> None:
        phyp = self.phyp # just for convenience
        # =================================================================
        # Step 1. Calculate the conormal at the contact line. 
        
        # project the discontinuous deformation gradient onto P1 to find the conormal vector m1
        dA = Measure(self.s_mesh, dim=1, order=5, coord_map=self.id_k) # the reference sheet mesh
        q_P1_k_basis = FunctionBasis(self.Q_P1_sp, dA)
        C_L2 = c_L2.assemble(q_P1_k_basis, q_P1_k_basis, dA)
        L_DQ = l_dq.assemble(q_P1_k_basis, dA, q_k = self.q_k._interpolate(dA))
        dq_k = Function(self.Q_P1_sp)
        dq_k[:] = spsolve(C_L2, L_DQ)

        # extract the conormal at the contact line
        dq_k_at_cl = dq_k.view(np.ndarray)[self.cl_dof_Q1].reshape(-1, 2) # type: np.ndarray # (-1, 2)
        dq_k_at_cl /= np.linalg.norm(dq_k_at_cl, ord=None, axis=1, keepdims=True) # (1, 2)
        # dq_k_at_cl[0] = -dq_k_at_cl[0]
        self.m1_k[:] = dq_k_at_cl.reshape(-1)

        # =================================================================
        # Step 2. Solve the fluid, the fluid-fluid interface, and the sheet deformation. 
        
        # set up the measures
        dx = Measure(self.mesh, dim=2, order=3)
        ds_4u = Measure(self.mesh, dim=1, order=5, tags=(3,)) # the fluid interface restricted from the bulk mesh
        da_4u = Measure(self.mesh, dim=1, order=5, tags=(4, 5)) # the deformed sheet restricted from the bulk mesh

        ds = Measure(self.i_mesh, dim=1, order=5)
        da = Measure(self.s_mesh, dim=1, order=5, coord_map=self.q_k_down) # the deformed sheet surface measure
        # dA = Measure(self.s_mesh, dim=1, order=5, coord_map=self.id_k)   # the reference sheet mesh at the last time step; defined already

        dp_i = Measure(self.i_mesh, dim=0, order=1, tags=(10,)) # the CL restricted from the fluid interface
        dp_s = Measure(self.s_mesh, dim=0, order=1, tags=(10,)) # the CL on the reference sheet mesh
        dp = Measure(self.cl_mesh, dim=0, order=1)

        # =================================================================
        # set up the function bases
        # fluid domain
        u_basis = FunctionBasis(self.U_sp, dx)
        p1_basis = FunctionBasis(self.P1_sp, dx)
        p0_basis = FunctionBasis(self.P0_sp, dx)
        # interface domain
        y_basis = FunctionBasis(self.Y_sp, ds)
        u_i_basis = FunctionBasis(self.U_sp, ds_4u)
        k_basis = FunctionBasis(self.K_sp, ds)
        # sheet domain
        u_da_basis = FunctionBasis(self.U_sp, da_4u)
        tau_basis = FunctionBasis(self.TAU_sp, da)
        q_da_basis = FunctionBasis(self.Q_sp, da)
        q_basis = FunctionBasis(self.Q_sp, dA)
        mom_basis = FunctionBasis(self.MOM_sp, dA)
        # CL from sheet
        y_cl_basis = FunctionBasis(self.Y_sp, dp_i)
        q_cl_basis = FunctionBasis(self.Q_sp, dp_s)
        m3_basis = FunctionBasis(self.M3_sp, dp)
        
        # =================================================================
        # assemble by blocks
        A_XIU = a_xiu.assemble(u_basis, u_basis, dx, eta=self.viscosity)
        A_XIP1 = a_xip.assemble(u_basis, p1_basis, dx)
        A_XIP0 = a_xip.assemble(u_basis, p0_basis, dx)
        A_XIK = a_xik.assemble(u_i_basis, k_basis, ds_4u)
        A_XITAU = c_L2.assemble(u_da_basis, tau_basis, da)

        A_ZY = a_zy.assemble(y_basis, y_basis, ds)
        A_ZK = a_zk.assemble(y_basis, k_basis, ds)
        A_ZM3 = a_zm3.assemble(y_cl_basis, m3_basis, dp_i)
        
        A_M3M3 = a_m3m3.assemble(m3_basis, m3_basis, dp, m1 = self.m1_k._interpolate(dp))
        A_M3Q = a_zm3.assemble(m3_basis, q_cl_basis, dp_s)
        L_M3_ADV = l_m3_adv.assemble(m3_basis, dp, dq_k=dq_k._interpolate(dp_s))

        B_PIQ = c_L2.assemble(tau_basis, q_da_basis, da)
        B_PIU = c_L2.assemble(tau_basis, u_da_basis, da)
        B_PITAU = b_pitau.assemble(tau_basis, tau_basis, da, mu_i=self.slip_fric)
        B_PIDQ = b_pidq.assemble(tau_basis, q_basis, da) # differentiate then push-forward

        C_PHIM = c_phim.assemble(q_basis, mom_basis, dA)
        C_PHIQ = c_phiq.assemble(q_basis, q_basis, dA, w_k = self.w_k._interpolate(dA))
        C_PHIQ_SURF = c_phiq_surf.assemble(q_da_basis, q_da_basis, da, gamma=self.surf_tens)
        C_WM = c_L2.assemble(mom_basis, mom_basis, dA)

        # collect the block matrices
        #    u,        p1,      p0,      tau,      y,       k,      m3,        w,      m
        A = bmat((
            (A_XIU,    -A_XIP1, -A_XIP0, A_XITAU,  None,    -phyp.gamma_3*A_XIK, None, None,   None),  # u
            (A_XIP1.T, None,    None,    None,     None,    None,   None,      None,   None),  # p1
            (A_XIP0.T, None,    None,    None,     None,    None,   None,      None,   None),  # p0
            (-solp.dt*B_PIU, None, None, solp.dt*B_PITAU,   None, None, None,  B_PIQ+solp.dt*phyp.U_adv*B_PIDQ, None),  # tau
            (None,     None,    None,    None,     A_ZY,    A_ZK,   -A_ZM3,    None,   None),  # y
            (-solp.dt*A_XIK.T,  None, None, None,  A_ZK.T,  None,   None,      None,   None),  # k
            (None,     None,    None,    None,     phyp.mu_cl/solp.dt*A_ZM3.T, None,   phyp.gamma_3*A_M3M3, -phyp.mu_cl/solp.dt*A_M3Q, None),  # m3
            (None,     None,    None,    -B_PIQ.T, None,    None,   phyp.gamma_3*A_M3Q.T, phyp.Y*C_PHIQ+C_PHIQ_SURF, -C_PHIM), # w
            (None,     None,    None,    None,     None,    None,   None,      C_PHIM.T, 1.0/phyp.B*C_WM), # m
        ), format="csr")
        # collect the right-hand-side
        self.u[:] = 0.0; self.p1[:] = 0.0; self.p0[:] = 0.0
        self.tau[:] = B_PIQ @ self.w_k - solp.dt*phyp.U_adv*(B_PIDQ @ self.id_k_lift)
        self.y[:] = 0.0; self.kappa[:] = A_ZK.T @ self.y_k
        self.m3[:] = (phyp.gamma_2 - phyp.gamma_1) * A_M3M3 @ self.m1_k \
            + phyp.mu_cl/solp.dt * (A_ZM3.T @ self.y_k - A_M3Q @ self.w_k) \
            + phyp.mu_cl * phyp.U_adv * L_M3_ADV
        self.w[:] = -C_PHIQ_SURF @ self.id_k_lift
        self.mom[:] = 0.0
        L = group_fn(self.u, self.p1, self.p0, self.tau, self.y, \
                     self.kappa, self.m3, self.w, self.mom)

        # homogeneize the right-hand-side
        self.tau[:] = 0.0
        self.y[self.cl_top_dof_Y] = (0.0, 0.75) # change this according to the mesh!
        self.kappa[:] = 0.0
        self.m3[:] = 0.0
        self.w[:] = 0.0
        self.mom[:] = 0.0
        sol_full = group_fn(self.u, self.p1, self.p0, self.tau, self.y, \
                            self.kappa, self.m3, self.w, self.mom)
        L = L - A @ sol_full

        # solve the coupled system
        free_dof = self.free_dof
        sol_free = spsolve(A[free_dof][:,free_dof], L[free_dof])
        sol_full[free_dof] = sol_free
        split_fn(sol_full, \
                 self.u, self.p1, self.p0, self.tau, self.y, \
                 self.kappa, self.m3, self.w, self.mom)
        q_star = self.w + self.id_k_lift

        # =================================================================
        # Step 3. Convert to FD form and find the upwind derivatives. 
        dq_fd = arrange_as_FD(self.Q_sp, lift_to_P2(self.Q_sp, dq_k)) # the projected dq
        q_fd = arrange_as_FD(self.Q_sp, self.q_k) # (x, 2)
        id_fd = arrange_as_FD(self.Q_sp, self.id_k_lift)[:,0] # (x,)
        dq_plus = np.zeros_like(q_fd)
        dq_minus = np.zeros_like(q_fd)
        dq_plus[:-1:2] = (-3*q_fd[:-1:2] + 4*q_fd[1::2] - q_fd[2::2]) / (id_fd[2::2] - id_fd[:-1:2])[:,np.newaxis]
        dq_minus[2::2] = (q_fd[:-1:2] - 4*q_fd[1::2] + 3*q_fd[2::2]) / (id_fd[2::2] - id_fd[:-1:2])[:,np.newaxis]
        dq_plus[1::2] = (q_fd[2::2] - q_fd[:-1:2]) / (id_fd[2::2] - id_fd[:-1:2])[:,np.newaxis]
        dq_minus[1::2] = dq_plus[1::2]

        dq_plus_cl = np.extract(self.cl_dof_fd >= 0, dq_plus).reshape(-1, 2)
        dq_minus_cl = np.extract(self.cl_dof_fd >= 0, dq_minus).reshape(-1, 2)

        # =================================================================
        # Step 4. Solve for the reference CL velocity and update the reference mesh. 

        # find the reference CL velocity
        slip_cl = (self.y.view(np.ndarray)[self.cl_dof_Y] \
            - q_star.view(np.ndarray)[self.cl_dof_Q2]) / solp.dt # type: np.ndarray # (2,)
        slip_cl = slip_cl.reshape(1, 2)
        m1_k_ = self.m1_k.view(np.ndarray).reshape(1, 2)
        dq_uw_cl = np.where((slip_cl[:,0] < 0)[:,np.newaxis], dq_minus_cl, dq_plus_cl) # (1,2)
        eta_cl = np.sum(slip_cl*m1_k_, axis=1) / np.sum(dq_uw_cl*m1_k_, axis=1) # (1,)

        # find the advected sheet
        refcl = self.refcl_hist[self.step] # (2,)
        eta = np.where(
            id_fd <= refcl[0], 
            (id_fd + 1.0) / (refcl[0] + 1.0) * eta_cl[0], # change this if the mesh changes!
            (1.0 - id_fd) / (1.0 - refcl[0]) * eta_cl[0]
        )
        dq_fd /= np.linalg.norm(dq_fd, axis=1, keepdims=True)
        adv = np.where((eta < 0)[:,np.newaxis], dq_minus, dq_plus) * eta[:,np.newaxis] # (x, 2)
        q_fd = arrange_as_FD(self.Q_sp, q_star) + solp.dt * np.sum(dq_fd * adv, axis=1, keepdims=True) * dq_fd
        
        eta = arrange_as_FE(self.Q_sp, np.vstack((eta, np.zeros_like(eta))).T)
        self.s_mesh.coord_map += down_to_P1(self.Q_P1_sp, eta) * solp.dt
        q = arrange_as_FE(self.Q_sp, q_fd)
        self.w = q - lift_to_P2(self.Q_sp, self.s_mesh.coord_map)

        # check the matching of the physical CL
        cl_diff = q.view(np.ndarray)[self.cl_dof_Q2] - self.y.view(np.ndarray)[self.cl_dof_Y]
        assert np.linalg.norm(cl_diff, ord=np.inf) < 1e-12
        
        # =================================================================
        # Step 5. Displace the bulk mesh and update all the meshes. 

        # update the interface mesh
        self.i_mesh.coord_map[:] = self.y

        # # no need to update the CL mesh as the coordinates are never used

        # solve for the bulk mesh displacement
        BMM_basis = FunctionBasis(self.mesh.coord_fe, dx) # for bulk mesh mapping
        A_EL = a_el.assemble(BMM_basis, BMM_basis, dx)
        # attach the displacement of the interface and the sheet
        BMM_int_dof = np.unique(self.mesh.coord_fe.getFacetDof(tags=(3,)))
        BMM_s_dof = np.unique(self.mesh.coord_fe.getFacetDof(tags=(4,5)))
        BMM_fix_dof = np.concatenate(self.mesh.coord_fe.getFacetDof(tags=(3,4,5,6,7,8,9)))
        bulk_disp = Function(self.mesh.coord_fe)
        bulk_disp[BMM_int_dof] = self.y - self.y_k
        bulk_disp[BMM_s_dof] = down_to_P1(self.Q_P1_sp, q - self.q_k)
        L_EL = Function(self.mesh.coord_fe)
        L_EL[:] = -A_EL @ bulk_disp # homogeneize the boundary conditions

        el_free_dof = group_dof((self.mesh.coord_fe,), (BMM_fix_dof,))
        bulk_disp[el_free_dof] = spsolve(A_EL[el_free_dof][:,el_free_dof], L_EL[el_free_dof])
        self.mesh.coord_map += bulk_disp

        print(Fore.GREEN + "\nt = {:.5f}, ".format((self.step+1) * self.solp.dt) + Style.RESET_ALL, end="")
        print("i-disp = {:.2e}, s-disp = {:.2e}, ".format(
            np.linalg.norm(self.y-self.y_k, ord=np.inf), np.linalg.norm(q-self.q_k, ord=np.inf)), end="")
        print("eta = {:+.2e}, ".format(eta_cl[0]), end="")
        print("|m| = {:.4f},, ".format(np.sqrt(self.m3[0]**2+self.m3[1]**2)), end="")

    def finish(self) -> None:
        super().finish()
        if self.args.vis:
            pyplot.ioff()
            pyplot.show()

# ===========================================================

if __name__ == "__main__":
    solp = SolverParameters(dt=1.0/256, Te=8.0)
    MCL_Runner(solp).run()
