from sys import argv
import numpy as np
from math import cos
from fem import *
from scipy import sparse
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
    Te: float = 1.0/1024 #1.0/8
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
        
    Y_fix_dof = np.unique(Y_fs.getFacetDof())
    # Y_bot_dof = np.unique(Y_fs.getCellDof(Measure(1, (4,5))))
    # Y_int_dof = np.unique(Y_fs.getCellDof(Measure(1, (3,))))

    Y_free_dof = group_dof((Y_fs,), (Y_fix_dof,))
    
    # get the piecewise constant viscosity and slip coefficient
    eta = np.where(mesh.cell_tag[2] == 1, phys.eta_1, 1.0)
    bot_flag = (mesh.cell_tag[1] == 5) | (mesh.cell_tag[1] == 4)
    bot_tag = mesh.cell_tag[1][bot_flag]
    beta = np.where(bot_tag == 5, phys.beta_1, 1.0)

    # initialize the measure and basis
    dx = Measure(mesh, 2, order=3)
    ds_i = Measure(mesh, 1, order=3, tags=(3, ))
    ds_bot = Measure(mesh, 1, order=3, tags=(4,5))
    ds = Measure(i_mesh, 1, order=3)
    dp = Measure(i_mesh, 0, order=1)

    # initialize the unknown
    u = Function(mixed_fs[0])
    p1 = Function(mixed_fs[1])
    p0 = Function(mixed_fs[2])
    x_m = Function(mixed_fs[3])
    x = Function(mixed_fs[3])
    kappa = Function(mixed_fs[4])

    Y = Function(Y_fs)
    Yg = np.zeros_like(Y)

    ax = None
    if solp.vis:
        pyplot.ion()
        ax = pyplot.subplot()

    m = solp.startStep
    while True:
        t = m * solp.dt
        if t >= solp.Te:
            break
        print("t = {0:.4f}".format(t))
        m += 1

        # visualization
        if solp.vis:
            ax.clear()
            # pyplot.tripcolor(mesh.coord_map[:,0], mesh.coord_map[:,1], facecolors=mesh.cell[2][:,-1], triangles=mesh.cell[2][:,:-1])
            # pyplot.triplot(mesh.coord_map[:,0], mesh.coord_map[:,1], triangles=mesh.cell[2][:,:-1])
            ax.axis("equal")
            pyplot.draw()
            pyplot.pause(1e-3)
            # maybe some more efficient ploting

        # first get the interface parametrization
        # np.copyto(x_m, i_mesh.coord_map)

        # [asm.updateGeometry() for asm in asms]

        # # assemble the coupled system
        # A_wu = asms[0].bilinear(Functional(a_wu, "grad"), eta=eta)
        # B_wp1 = asms[1].bilinear(Functional(b_wp))
        # B_wp0 = asms[2].bilinear(Functional(b_wp))
        # A_wk = asms[3].bilinear(Functional(a_wk, "f"))
        # S_wu = asms[4].bilinear(Functional(a_slip_wu, "f"), beta=beta)

        # A_gx = asms[5].bilinear(Functional(a_gx, "grad"))
        # A_gk = asms[6].bilinear(Functional(a_wk, "f"))
        # A_psiu = asms[7].bilinear(Functional(a_psiu, "f"))
        # S_gx = asms[8].bilinear(Functional(a_slip_gx, "f"))

        # A = sparse.bmat(((A_wu + 1.0/phys.l_s*S_wu, -B_wp1, -B_wp0, None, -1.0/phys.Ca*A_wk), 
        #                 (B_wp1.T, None, None, None, None), 
        #                 (B_wp0.T, None, None, None, None), 
        #                 (None, None, None, A_gx + phys.beta_s*phys.Ca/solp.dt*S_gx, A_gk), 
        #                 (-solp.dt*A_psiu, None, None, A_gk.T, None)), 
        #                 format="csr")
        
        # # assemble the RHS
        # L_g = asms[8].linear(Functional(l_g, "f"))
        # L_g_x = asms[8].linear(Functional(l_slip_g_x, "f"), x_m=x_m)
        # L_psi_x = asms[7].linear(Functional(l_psi_x, "f"), x_m=x_m)

        # u[:] = 0.0
        # p1[:] = 0.0
        # p0[:] = 0.0
        # L = group_fn(u, p1, p0, phys.cosY*L_g + phys.beta_s*phys.Ca/solp.dt*L_g_x, L_psi_x)

        # # Since the essential conditions are all homogeneous,
        # # we don't need to homogeneize the system
        # sol_vec = np.zeros_like(L)
        
        # # solve the linear system
        # sol_vec_free = spsolve(A[free_dof][:,free_dof], L[free_dof])
        # sol_vec[free_dof] = sol_vec_free
        # split_fn(sol_vec, u, p1, p0, x, kappa)
        # u.update()
        # p1.update()

        # # some useful info ...
        # i_disp = x - x_m
        # # print("Interface displacement = {}".format(np.linalg.norm(i_disp.reshape(-1), np.inf)))

        # # solve the displacement on the substrate
        # cl_dof = cl_dof.reshape(-1)
        # if x[cl_dof[0], 0] > x[cl_dof[1], 0]: # make sure [0] on the left, [1] on the right
        #     cl_dof = cl_dof[::-1]
        # # set up the dirichlet condition
        # Yg[:] = 0.0
        # Y0_m = mesh.coord_map[Y_bot_dof, 0] # the x coordinate of the grid points on the substrate
        # cl_pos = (x_m[cl_dof[0], 0], x_m[cl_dof[1], 0])
        # Yg[Y_bot_dof, 0] = np.where(
        #     Y0_m < cl_pos[0], (Y0_m + 1.0) / (cl_pos[0] + 1.0) * i_disp[cl_dof[0],0], 
        #     np.where(Y0_m > cl_pos[1], (1.0 - Y0_m) / (1.0 - cl_pos[1]) * i_disp[cl_dof[1],0], 
        #              (i_disp[cl_dof[0],0] * (cl_pos[1] - Y0_m) + i_disp[cl_dof[1],0] * (Y0_m - cl_pos[0])) / (cl_pos[1] - cl_pos[0]))
        # )
        # Yg[Y_int_dof] = i_disp

        # # solve the linear elastic equation for the bulk mesh deformation ...
        # A_el = asms[9].bilinear(Functional(a_el, "grad"))
        # L_el = -A_el @ Yg.reshape(-1)
        # sol_vec = Yg.copy().reshape(-1)
        # sol_vec_free = spsolve(A_el[Y_free_dof][:,Y_free_dof], L_el[Y_free_dof])
        # sol_vec[Y_free_dof] = sol_vec_free
        # split_fn(sol_vec, Y)

        # # move the mesh
        # Y += mesh.coord_map
        # np.copyto(mesh.coord_map, Y)
        # np.copyto(i_mesh.coord_map, x)
    # end time loop
