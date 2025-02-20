import numpy as np
from math import cos
from fem import *
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from runner import *
from matplotlib import pyplot

# physical groups from GMSH
# group_name = {"fluid_1": 1, "fluid_2": 2, "interface": 3, "dry": 4, "wet": 5, \
#              "right": 6, "top": 7, "left": 8, "cl": 9}

@BilinearForm
def a_wu(w, u, x, _, eta) -> np.ndarray:
    # eta: (Ne,)
    # grad: (2, 2, Ne, Nq)    
    z = np.zeros(x.shape[1:]) # (Ne, Nq)
    for i, j in (0,0), (0,1), (1,0), (1,1):
        z += (u.grad[i,j] + u.grad[j,i]) * w.grad[i,j]
    return (z * eta[:, np.newaxis])[np.newaxis] * x.dx

@BilinearForm
def b_wp(w, p, x, _) -> np.ndarray:
    # w.grad: (2,2,Ne,Nq)
    # p: (1, 1, Nq)
    z = (w.grad[0,0] + w.grad[1,1]) * p[0]
    return z[np.newaxis] * x.dx

@BilinearForm
def a_wk(w, kappa, x, _) -> np.ndarray:
    # kappa: (1, 1, Nq)
    # w: (2, Nf, Nq)
    # x.fn: (2, Nf, Nq)
    return np.sum(x.fn * w, axis=0, keepdims=True) * kappa * x.ds # (2, Nf, Nq)

@BilinearForm
def a_slip_wu(w, u, x, _, beta) -> np.ndarray:
    # u, w: (2, Nf, Nq)
    # beta: (Nf, )
    return (u[0] * w[0] * beta[:, np.newaxis])[np.newaxis] * x.ds

@BilinearForm
def a_gx(g, x, z, _) -> np.ndarray:
    # grad: (2, 2, Ne, Nq)
    return np.sum(g.grad * x.grad, axis=(0,1))[np.newaxis] * z.dx

@BilinearForm
def a_gk(g, kappa, z, _) -> np.ndarray:
    # kappa: (1, 1, Nq)
    # g: (2, Nf, Nq)
    # z.cn: (2, Nf, Nq)
    return np.sum(g * z.cn, axis=0, keepdims=True) * kappa * z.dx

@BilinearForm
def a_psiu(psi, u, z, _) -> np.ndarray:
    # u: (2, Nf, Nq)
    # psi: (1, 1, Nq)
    # z.cn: (2, Nf, Nq)
    return np.sum(u * z.cn, axis=0, keepdims=True) * psi * z.dx

@BilinearForm
def a_slip_gx(g, x, z, _) -> np.ndarray:
    # g: (2, 1, Nq)
    # x: (2, 1, Nq)
    # z.ds (1, 2, Nq)
    return (g[0] * x[0])[np.newaxis] * z.ds

@LinearForm
def l_g(g, z) -> np.ndarray:
    # z: (2, 2, Nq)
    # g: (2, 2, Nq)
    r = np.where(z[0] > 0.0, g[0], -g[0]) # (2, Nq)
    return r[np.newaxis] * z.ds

# for linear elasticity
@BilinearForm
def a_el(Z, Y, x, _) -> np.ndarray:
    # grad: (2, 2, Ne, Nq)
    # x.dx: (1, Ne, Nq)
    lam_dx = x.dx + (x.dx.max() - x.dx.min()) # (1, Ne, Nq)
    r = np.zeros(x.shape[1:]) # (Ne,Nq)
    tr = Y.grad[0,0] + Y.grad[1,1] # (Ne, Nq)
    for i, j in (0,0), (0,1), (1,0), (1,1):
        r += (Y.grad[i,j] + Y.grad[j,i] + (i==j) * tr) * Z.grad[i,j]
    return r[np.newaxis] * lam_dx


class PhysicalParameters:
    eta_1: float = 1.0
    eta_2: float = 0.1
    beta_1: float = 1e3
    beta_s: float = 1.
    Ca: float = 0.1
    cosY: float = cos(np.pi/3)

class TwoPhaseStokes(Runner):
    def prepare(self, phyp: PhysicalParameters) -> None:
        super().prepare()
        
        self.phyp = phyp
        with open(self._get_output_name("PhysicalParameters"), "wb") as f:
            pickle.dump(self.phyp, f)
        # load the meshes
        # physical groups from GMSH
        # group_name = {"fluid_1": 1, "fluid_2": 2, "interface": 3, "dry": 4, "wet": 5, \
        #              "right": 6, "top": 7, "left": 8, "cl": 9, "clamp": 10}
        self.mesh = Mesh()
        self.mesh.load(self.args.mesh_name)
        for _ in range(self.args.spaceref):
            self.mesh = splitRefine(self.mesh)
        setMeshMapping(self.mesh)    
        self.i_mesh = self.mesh.view(1, tags=(3, ))
        setMeshMapping(self.i_mesh)

        # get the piecewise constant viscosity and slip coefficient
        self.viscosity = np.where(self.mesh.cell_tag[2] == 1, phyp.eta_1, phyp.eta_2)
        bot_flag = (self.mesh.cell_tag[1] == 5) | (self.mesh.cell_tag[1] == 4)
        bot_tag = self.mesh.cell_tag[1][bot_flag]
        self.slip_fric = np.where(bot_tag == 5, phyp.beta_1, phyp.beta_1)

        def periodic_constraint(x: np.ndarray) -> np.ndarray:
            flag = np.abs(x[:,0] - 1.0) < 1e-12
            x[flag, 0] -= 2.0

        # set up the function spaces
        self.mixed_fs = (
            FunctionSpace(self.mesh, VectorElement(TriP2, num_copy=2), constraint=periodic_constraint), # U
            FunctionSpace(self.mesh, TriP1, constraint=periodic_constraint), # P1
            FunctionSpace(self.mesh, TriDG0, constraint=periodic_constraint), # P0
            self.i_mesh.coord_fe, # X
            FunctionSpace(self.i_mesh, LineP1), # K
        )

        # extract the useful DOFs
        self.top_dof = np.where(self.mixed_fs[0].dof_loc[:,1] > 1-1e-12)[0]
        self.bot_dof = np.where(self.mixed_fs[0].dof_loc[:,1] < 1e-12)[0]
        cl_pos = self.i_mesh.point[self.i_mesh.point_tag == 9, 0]
        assert cl_pos.size == 2
        flag = (self.mixed_fs[3].dof_loc[:,1] < 1e-12) & \
            ((np.abs(self.mixed_fs[3].dof_loc[:,0] - cl_pos[0]) < 1e-12) | (np.abs(self.mixed_fs[3].dof_loc[:,0] - cl_pos[1]) < 1e-12))
        self.cl_dof = np.where(flag)[0]
        assert self.cl_dof.size == 4

        self.free_dof = group_dof(self.mixed_fs, (np.concatenate((self.top_dof, self.bot_dof[1::2])), np.array((0,)), np.array((0,)), self.cl_dof[1::2], None))
            
        Y_fs = self.mesh.coord_fe # type: FunctionSpace
        self.Y_bot_dof = np.where(Y_fs.dof_loc[:,1] < 1e-12)[0]
        self.Y_int_dof = Y_fs.getDofByLocation(self.i_mesh.coord_fe.dof_loc[::2])
        Y_bound_dof = np.where((Y_fs.dof_loc[:,1] > 1-1e-12) | (Y_fs.dof_loc[:,0] < -1+1e-12) | (Y_fs.dof_loc[:,0] > 1-1e-12))[0]
        Y_fix_dof = np.unique(np.concatenate((self.Y_bot_dof, self.Y_int_dof, Y_bound_dof)))

        self.Y_free_dof = group_dof((Y_fs, ), (Y_fix_dof, ))

        # allocate the functions
        self.u = Function(self.mixed_fs[0])
        self.p1 = Function(self.mixed_fs[1])
        self.p0 = Function(self.mixed_fs[2])
        self.x = Function(self.mixed_fs[3])
        self.kappa = Function(self.mixed_fs[4])

        self.Y = Function(Y_fs) # the mesh deformation
    
        # read checkpoints from file
        if self.args.resume:
            self.mesh.coord_map[:] = self.resume_file["bulk_coord_map"]
            self.i_mesh.coord_map[:] = self.resume_file["x_m"]
            del self.resume_file

        # prepare visualization
        if self.args.vis:
            pyplot.ion()
            self.ax = pyplot.subplot()
            self.triangles = Y_fs.elem_dof[::2].T.copy()
            assert np.all(self.triangles % 2 == 0)
            self.triangles //= 2

    def pre_step(self) -> bool:
        if self.args.vis:
            self.ax.clear()
            # press = p0.view(np.ndarray)[mixed_fs[2].elem_dof][0] + np.sum(p1.view(np.ndarray)[mixed_fs[1].elem_dof], axis=0) / 3 # (Nt, )
            # tpc = ax.tripcolor(mesh.coord_map[::2], mesh.coord_map[1::2], press, triangles=triangles)
            # if colorbar is None:
            #     colorbar = pyplot.colorbar(tpc)
            # else:
            #     colorbar.update_normal(tpc)
            self.ax.tripcolor(self.mesh.coord_map[::2], self.mesh.coord_map[1::2], self.mesh.cell_tag[2], triangles=self.triangles)
            # plot the velocity
            _u = self.u.view(np.ndarray); _n = self.mesh.coord_map.size
            self.ax.quiver(self.mesh.coord_map[::2], self.mesh.coord_map[1::2], _u[:_n:2], _u[1:_n:2])
            self.ax.triplot(self.mesh.coord_map[::2], self.mesh.coord_map[1::2], triangles=self.triangles)
            self.ax.axis("equal")
            pyplot.draw()
            pyplot.pause(1e-4)
        if self.step % self.solp.stride_checkpoint == 0:
            filename = self._get_output_name("{:04d}.npz".format(self.step))
            np.savez(filename, bulk_coord_map=self.mesh.coord_map, x_m=self.i_mesh.coord_map)
            print("\n* Checkpoint saved to " + filename)
        return self.step >= self.num_steps
    
    def main_step(self) -> None:
        t = self.step * self.solp.dt

        # initialize the measure and basis
        dx = Measure(self.mesh, 2, order=3)
        ds_i = Measure(self.mesh, 1, order=3, tags=(3, ))
        ds_bot = Measure(self.mesh, 1, order=3, tags=(4, 5))
        ds = Measure(self.i_mesh, 1, order=3)
        dp = Measure(self.i_mesh, 0, order=1, tags=(9, ))

        u_basis = FunctionBasis(self.mixed_fs[0], dx)
        p1_basis = FunctionBasis(self.mixed_fs[1], dx)
        p0_basis = FunctionBasis(self.mixed_fs[2], dx)
        u_i_basis = FunctionBasis(self.mixed_fs[0], ds_i)
        u_bot_basis = FunctionBasis(self.mixed_fs[0], ds_bot)
        x_basis = FunctionBasis(self.mixed_fs[3], ds)
        k_basis = FunctionBasis(self.mixed_fs[4], ds)
        x_cl_basis = FunctionBasis(self.mixed_fs[3], dp)

        # save the interface parametrization
        x_m = Function(self.mixed_fs[3])
        np.copyto(x_m, self.i_mesh.coord_map) 

        # assemble the coupled system
        A_wu = a_wu.assemble(u_basis, u_basis, None, None, eta=self.viscosity)
        B_wp1 = b_wp.assemble(u_basis, p1_basis)
        B_wp0 = b_wp.assemble(u_basis, p0_basis)
        A_wk = a_wk.assemble(u_i_basis, k_basis)
        S_wu = a_slip_wu.assemble(u_bot_basis, u_bot_basis, None, None, beta=self.slip_fric)

        A_gx = a_gx.assemble(x_basis, x_basis)
        A_gk = a_gk.assemble(x_basis, k_basis)
        A_psiu = a_psiu.assemble(k_basis, u_i_basis)
        S_gx = a_slip_gx.assemble(x_cl_basis, x_cl_basis)
        
        phyp = self.phyp
        A = bmat(((A_wu + S_wu, -B_wp1, -B_wp0, None, -1.0/phyp.Ca*A_wk), 
                (B_wp1.T, None, None, None, None), 
                (B_wp0.T, None, None, None, None), 
                (None, None, None, A_gx + phyp.beta_s*phyp.Ca/solp.dt*S_gx, A_gk), 
                (-solp.dt*A_psiu, None, None, A_gk.T, None)), 
                format="csc")
        
        # assemble the RHS
        L_g = phyp.cosY * l_g.assemble(x_cl_basis)
        self.u[:] = 0.0
        self.p1[:] = 0.0
        self.p0[:] = 0.0
        L = group_fn(self.u, self.p1, self.p0, \
                     L_g + phyp.beta_s*phyp.Ca/solp.dt*(S_gx @ x_m), A_gk.T @ x_m)

        # Since the essential conditions are all homogeneous,
        # we don't need to homogeneize the system
        sol_vec = np.zeros_like(L)
        
        # solve the linear system
        free_dof = self.free_dof
        sol_vec_free = spsolve(A[free_dof][:,free_dof], L[free_dof])
        sol_vec[free_dof] = sol_vec_free
        split_fn(sol_vec, self.u, self.p1, self.p0, self.x, self.kappa)

        # some useful info ...
        i_disp = self.x - x_m
        print("t = {:.4f}, disp = {:.2e}".format(t, np.linalg.norm(i_disp, np.inf)))

        # solve the displacement on the substrate 
        self.Y[:] = 0.0
        Y0_m = self.mesh.coord_map[self.Y_bot_dof[::2]] # the x coordinate of the grid points on the substrate
        cl_pos = x_m[self.cl_dof[::2]] # [0] for the left, [1] for the right
        cl_disp = i_disp[self.cl_dof[::2]] # same as above
        assert cl_pos[0] < cl_pos[1]
        self.Y[self.Y_bot_dof[::2]] = np.where(
            Y0_m <= cl_pos[0], (Y0_m + 1.0) / (cl_pos[0] + 1.0) * cl_disp[0], 
            np.where(Y0_m >= cl_pos[1], (1.0 - Y0_m) / (1.0 - cl_pos[1]) * cl_disp[1], 
                    (cl_disp[0] * (cl_pos[1] - Y0_m) + cl_disp[1] * (Y0_m - cl_pos[0])) / (cl_pos[1] - cl_pos[0]))
        )
        self.Y[self.Y_int_dof] = i_disp

        # solve the linear elastic equation for the bulk mesh deformation 
        Y_basis = FunctionBasis(self.mesh.coord_fe, dx)
        A_el = a_el.assemble(Y_basis, Y_basis)
        L_el = -A_el @ self.Y
        Y_free_dof = self.Y_free_dof
        sol_vec_free = spsolve(A_el[Y_free_dof][:,Y_free_dof], L_el[Y_free_dof])
        self.Y[Y_free_dof] = sol_vec_free

        # move the mesh
        self.Y += self.mesh.coord_map
        np.copyto(self.mesh.coord_map, self.Y)
        np.copyto(self.i_mesh.coord_map, self.x)

    def finish(self) -> None:
        super().finish()
        if self.args.vis:
            pyplot.ioff()
            pyplot.show()


if __name__ == "__main__":
    solp = SolverParameters(dt = 1.0/256, Te = 1.0)
    phyp = PhysicalParameters()
    solver = TwoPhaseStokes(solp)
    solver.prepare(phyp)
    solver.run()
    solver.finish()
