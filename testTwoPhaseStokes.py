from sys import argv
import numpy as np
from math import cos
from fem import *
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot

# physical groups from GMSH
# group_name = {"fluid_1": 1, "fluid_2": 2, "interface": 3, "dry": 4, "wet": 5, \
#              "right": 6, "top": 7, "left": 8, "cl": 9}

@BilinearForm
def a_wu(w, u, x, eta) -> np.ndarray:
    # eta: (Ne,)
    # grad: (2, 2, Ne, Nq)    
    z = np.zeros(x.shape[1:]) # (Ne, Nq)
    for i, j in (0,0), (0,1), (1,0), (1,1):
        z += (u.grad[i,j] + u.grad[j,i]) * w.grad[i,j]
    return (z * eta[:, np.newaxis])[np.newaxis] * x.dx

@BilinearForm
def b_wp(w, p, x) -> np.ndarray:
    # w.grad: (2,2,Ne,Nq)
    # p: (1, 1, Nq)
    z = (w.grad[0,0] + w.grad[1,1]) * p[0]
    return z[np.newaxis] * x.dx

@BilinearForm
def a_wk(w, kappa, x) -> np.ndarray:
    # kappa: (1, 1, Nq)
    # w: (2, 1, Nq)
    # x.fn: (2, Nf, Nq)
    return np.sum(x.fn * w, axis=0, keepdims=True) * kappa * x.ds # (2, Nf, Nq)

@BilinearForm
def a_slip_wu(w, u, x, beta) -> np.ndarray:
    # u, w: (2, 1, Nq)
    # beta: (Nf, )
    return (u[0] * w[0] * beta[:, np.newaxis])[np.newaxis] * x.ds

@BilinearForm
def a_gx(g, x, z) -> np.ndarray:
    # grad: (2, 2, Ne, Nq)
    return np.sum(g.grad * x.grad, axis=(0,1))[np.newaxis] * z.dx

@BilinearForm
def a_gk(g, kappa, z) -> np.ndarray:
    # kappa: (1, 1, Nq)
    # g: (2, Nf, Nq)
    # z.cn: (2, Nf, Nq)
    return np.sum(g * z.cn, axis=0, keepdims=True) * kappa * z.dx

@BilinearForm
def a_psiu(psi, u, z) -> np.ndarray:
    # u: (2, 1, Nq)
    # psi: (1, 1, Nq)
    # z.fn: (2, Ne, Nq)
    return np.sum(u * z.fn, axis=0, keepdims=True) * psi * z.ds

@BilinearForm
def a_slip_gx(g, x, z) -> np.ndarray:
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
def a_el(Z, Y, x) -> np.ndarray:
    # grad: (2, 2, Ne, Nq)
    # x.dx: (1, Ne, Nq)
    lam_dx = x.dx + (x.dx.max() - x.dx.min()) # (1, Ne, Nq)
    r = np.zeros(x.shape[1:]) # (Ne,Nq)
    tr = Y.grad[0,0] + Y.grad[1,1] # (Ne, Nq)
    for i, j in (0,0), (0,1), (1,0), (1,1):
        r += (Y.grad[i,j] + Y.grad[j,i] + (i==j) * tr) * Z.grad[i,j]
    return r[np.newaxis] * lam_dx


class PhysicalParameters:
    eta_1: float = 10.0
    beta_1: float = 0.1
    beta_s: float = 0.1
    l_s: float = 0.1
    Ca: float = 0.01
    cosY: float = cos(np.pi*2.0/3)

class SolverParemeters:
    dt: float = 1.0/1024
    Te: float = 1.0/4
    startStep: int = 0
    stride: int = 1
    numChekpoint: int = 0
    vis: bool = True


if __name__ == "__main__":

    phys = PhysicalParameters()
    solp = SolverParemeters()
    solp.vis = len(argv) >= 2 and bool(argv[1])

    mesh = Mesh()
    mesh.load("mesh/two-phase.msh")
    setMeshMapping(mesh)
    def periodic_constraint(x: np.ndarray) -> np.ndarray:
        flag = np.abs(x[:,0] - 1.0) < 1e-12
        x[flag, 0] -= 2.0
    
    i_mesh = mesh.view(1, (3, ))
    setMeshMapping(i_mesh)

    mixed_fs = (
        FunctionSpace(mesh, VectorElement(TriP2, 2), constraint=periodic_constraint), # U
        FunctionSpace(mesh, TriP1, constraint=periodic_constraint), # P1
        FunctionSpace(mesh, TriDG0, constraint=periodic_constraint), # P0
        i_mesh.coord_fe, # X
        FunctionSpace(i_mesh, LineP1), # K
    )
    
    Y_fs = mesh.coord_fe # should be VectorElement(TriP1, 2)

    # determine the fixed dof
    top_dof = np.unique(mixed_fs[0].getFacetDof((7, )))
    bot_dof = np.unique(mixed_fs[0].getFacetDof((4, 5)))
    bot_vert_dof = bot_dof[1::2]
    cl_dof = np.unique(mixed_fs[3].getFacetDof((9, )))
    cl_vert_dof = cl_dof[1::2]

    free_dof = group_dof(mixed_fs, (np.concatenate((top_dof, bot_vert_dof)), np.array((0,)), np.array((0,)), cl_vert_dof, None))
        
    Y_fix_dof = np.unique(Y_fs.getFacetDof((3, 4, 5, 6, 7, 8)))
    Y_bot_dof = np.unique(Y_fs.getFacetDof((4, 5)))
    Y_int_dof = np.unique(Y_fs.getFacetDof((3,)))

    Y_free_dof = group_dof((Y_fs,), (Y_fix_dof,))
    
    # get the piecewise constant viscosity and slip coefficient
    eta = np.where(mesh.cell_tag[2] == 1, phys.eta_1, 1.0)
    bot_flag = (mesh.cell_tag[1] == 5) | (mesh.cell_tag[1] == 4)
    bot_tag = mesh.cell_tag[1][bot_flag]
    beta = np.where(bot_tag == 5, phys.beta_1, 1.0)

    # initialize the unknowns
    u = Function(mixed_fs[0])
    p1 = Function(mixed_fs[1])
    p0 = Function(mixed_fs[2])
    x_m = Function(mixed_fs[3])
    x = Function(mixed_fs[3])
    kappa = Function(mixed_fs[4])

    Y = Function(Y_fs)
    # Yg = np.zeros_like(Y)

    if solp.vis:
        pyplot.ion()
        ax = pyplot.subplot()
        # nv = NodeVisualizer(mesh, Y_fs)
        triangles = Y_fs.elem_dof[::2, :].T
        assert np.all(triangles % 2 == 0)
        triangles = triangles // 2

    m = solp.startStep
    while True:
        t = m * solp.dt
        if t >= solp.Te:
            break
        print("Solving t = {0:.4f}, ".format(t), end="")
        m += 1

        # visualization
        if solp.vis:
            ax.clear()
            pyplot.tripcolor(mesh.coord_map[::2], mesh.coord_map[1::2], mesh.cell_tag[2], triangles=triangles)
            pyplot.triplot(mesh.coord_map[::2], mesh.coord_map[1::2], triangles=triangles)
            ax.axis("equal")
            pyplot.draw()
            pyplot.pause(1e-3)
            
        # initialize the measure and basis
        dx = Measure(mesh, 2, order=3)
        ds_i = Measure(mesh, 1, order=3, tags=(3, ))
        ds_bot = Measure(mesh, 1, order=3, tags=(4,5))
        ds = Measure(i_mesh, 1, order=3)
        dp = Measure(i_mesh, 0, order=1, tags=(9, ))

        u_basis = FunctionBasis(mixed_fs[0], dx)
        p1_basis = FunctionBasis(mixed_fs[1], dx)
        p0_basis = FunctionBasis(mixed_fs[2], dx)
        u_i_basis = FunctionBasis(mixed_fs[0], ds_i)
        u_bot_basis = FunctionBasis(mixed_fs[0], ds_bot)

        x_basis = FunctionBasis(mixed_fs[3], ds)
        k_basis = FunctionBasis(mixed_fs[4], ds)
        x_cl_basis = FunctionBasis(mixed_fs[3], dp)

        # first get the interface parametrization
        np.copyto(x_m, i_mesh.coord_map)

        # assemble the coupled system
        A_wu = a_wu.assemble(u_basis, u_basis, dx, eta=eta)
        B_wp1 = b_wp.assemble(u_basis, p1_basis, dx)
        B_wp0 = b_wp.assemble(u_basis, p0_basis, dx)
        A_wk = a_wk.assemble(u_i_basis, k_basis, ds_i)
        S_wu = a_slip_wu.assemble(u_bot_basis, u_bot_basis, ds_bot, beta=beta)

        A_gx = a_gx.assemble(x_basis, x_basis, ds)
        A_gk = a_gk.assemble(x_basis, k_basis, ds)
        A_psiu = a_psiu.assemble(k_basis, u_i_basis, ds_i)
        S_gx = a_slip_gx.assemble(x_cl_basis, x_cl_basis, dp)

        A = bmat(((A_wu + 1.0/phys.l_s*S_wu, -B_wp1, -B_wp0, None, -1.0/phys.Ca*A_wk), 
                (B_wp1.T, None, None, None, None), 
                (B_wp0.T, None, None, None, None), 
                (None, None, None, A_gx + phys.beta_s*phys.Ca/solp.dt*S_gx, A_gk), 
                (-solp.dt*A_psiu, None, None, A_gk.T, None)), 
                format="csr")
        
        # assemble the RHS
        L_g = phys.cosY * l_g.assemble(x_cl_basis, dp)
        u[:] = 0.0
        p1[:] = 0.0
        p0[:] = 0.0
        L = group_fn(u, p1, p0, L_g + phys.beta_s*phys.Ca/solp.dt*(S_gx @ x_m), A_gk.T @ x_m)

        # Since the essential conditions are all homogeneous,
        # we don't need to homogeneize the system
        sol_vec = np.zeros_like(L)
        
        # solve the linear system
        sol_vec_free = spsolve(A[free_dof][:,free_dof], L[free_dof])
        sol_vec[free_dof] = sol_vec_free
        split_fn(sol_vec, u, p1, p0, x, kappa)

        # some useful info ...
        i_disp = x - x_m
        print("displacement = {0:.2e}".format(np.linalg.norm(i_disp, np.inf)))

        # solve the displacement on the substrate 
        Y[:] = 0.0
        Y0_m = mesh.coord_map[Y_bot_dof[::2]] # the x coordinate of the grid points on the substrate
        cl_pos = x_m[cl_dof[::2]] # [0] for the left, [1] for the right
        cl_disp = i_disp[cl_dof[::2]] # save as above
        assert cl_pos[0] < cl_pos[1]
        Y[Y_bot_dof[::2]] = np.where(
            Y0_m <= cl_pos[0], (Y0_m + 1.0) / (cl_pos[0] + 1.0) * cl_disp[0], 
            np.where(Y0_m >= cl_pos[1], (1.0 - Y0_m) / (1.0 - cl_pos[1]) * cl_disp[1], 
                     (cl_disp[0] * (cl_pos[1] - Y0_m) + cl_disp[1] * (Y0_m - cl_pos[0])) / (cl_pos[1] - cl_pos[0]))
        )
        Y[Y_int_dof] = i_disp

        # solve the linear elastic equation for the bulk mesh deformation 
        Y_basis = FunctionBasis(Y_fs, dx)
        A_el = a_el.assemble(Y_basis, Y_basis, dx)
        L_el = -A_el @ Y
        sol_vec_free = spsolve(A_el[Y_free_dof][:,Y_free_dof], L_el[Y_free_dof])
        Y[Y_free_dof] = sol_vec_free

        # move the mesh
        Y += mesh.coord_map
        np.copyto(mesh.coord_map, Y)
        np.copyto(i_mesh.coord_map, x)
    # end time loop

    print("Finished.")
    pyplot.ioff()
    pyplot.show()
