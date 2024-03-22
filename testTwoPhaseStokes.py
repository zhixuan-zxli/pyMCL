import numpy as np
from math import cos
from mesh import Mesh, Measure
from mesh_util import splitRefine, setMeshMapping
from fe import FiniteElement, TriDG0, TriP1, TriP2, LineP1, group_dof
from function import Function, split_fn, group_fn
from assemble import assembler, Form
from scipy import sparse
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot

# physical groups from GMSH
# group_name = {"fluid_1": 1, "fluid_2": 2, "interface": 3, "dry": 4, "wet": 5, \
#              "right": 6, "top": 7, "left": 8, "cl": 9}

def a_wu(w, u, coord, eta) -> np.ndarray:
    # eta: (Ne,)
    # grad: (2, 2, Ne, Nq)
    a = np.zeros((2, 2, coord.shape[1], coord.shape[2]))
    a[0,0] = 2.0 * w.grad[0,0] * u.grad[0,0] + w.grad[0,1] * u.grad[0,1]
    a[0,1] = w.grad[0,1] * u.grad[1,0]
    a[1,0] = w.grad[1,0] * u.grad[0,1]
    a[1,1] = w.grad[1,0] * u.grad[1,0] + 2.0 * w.grad[1,1] * u.grad[1,1]
    return a * eta.reshape(1, 1, -1, 1) * coord.dx[np.newaxis]

def b_wp(w, p, coord) -> np.ndarray:
    # w.grad: (2,2,Ne,Nq)
    # p: (1, 1, Nq)
    b = np.zeros((2, 1, coord.shape[1], coord.shape[2]))
    b[0,0,:,:] = w.grad[0,0,:,:] * p[0,:,:]
    b[1,0,:,:] = w.grad[1,1,:,:] * p[0,:,:]
    return b * coord.dx[np.newaxis]

def a_wk(w, kappa, coord) -> np.ndarray:
    # kappa: (1, 1, Nq)
    # w: (2, 1, Nq)
    # coord.n: (2, Ne, Nq)
    # output: (2, 1, Ne, Nq)
    a = coord.n * w * kappa * coord.dx # (2, Ne, Nq)
    return a[:, np.newaxis]

def a_slip_wu(w, u, coord, beta) -> np.ndarray:
    # u, w: (2, 1, Nq)
    # beta: (Ne, )
    a = np.zeros((2, 2, coord.shape[1], coord.shape[2]))
    a[0,0] = u[0] * w[0] * beta[:,np.newaxis] * coord.dx[0] # (Ne, Nq)
    return a

def a_gx(g, x, coord) -> np.ndarray:
    # grad: (2, 2, Ne, Nq)
    a = np.zeros((2, 2, coord.shape[1], coord.shape[2]))
    a[0,0] = np.sum(x.grad[0] * g.grad[0], axis=0) * coord.dx[0]
    a[1,1] = np.sum(x.grad[1] * g.grad[1], axis=0) * coord.dx[0]
    return a

# def a_gk(g, kappa, coord) -> np.ndarray:
#     return a_wk(g, kappa, coord)

def a_psiu(psi, u, coord) -> np.ndarray:
    # u: (2, 1, Nq)
    # psi: (1, 1, Nq)
    # coord.n: (2, Ne, Nq)
    a = u * coord.n * psi * coord.dx # (2, Ne, Nq)
    return a[np.newaxis, :]

# def a_psiu(psi, u, coord) -> np.ndarray:
#     return a_psix(psi, u, coord)

def l_psi_x(psi, coord, x_m) -> np.ndarray:
    # x_m: (2, Ne, Nq)
    # coord.n: (2, Ne, Nq)
    # psi: (1, 1, Nq)
    return np.sum(x_m * coord.n, axis=0, keepdims=True) * psi * coord.dx

def a_slip_gx(g, x, coord) -> np.ndarray:
    # g: (2, 1, Nq)
    # x: (2, 1, Nq)
    a = np.zeros((2, 2, coord.shape[1], coord.shape[2]))
    a[0,0] = g[0] * x[0]
    return a

def l_slip_g_x(g, coord, x_m) -> np.ndarray:
    # g: (2, 1, Nq)
    # x_m: (2, Ne, Nq)
    l = np.zeros((2, coord.shape[1], coord.shape[2]))
    l[0] = g[0] * x_m[0] # (Ne, Nq)
    return l

def l_g(g, coord) -> np.ndarray:
    # coord: (2, Ne, Nq)
    # g: (2, 1, Nq)
    l = np.zeros((2, coord.shape[1], coord.shape[2]))
    l[0] = np.where(coord[0] > 0, g[0], -g[0]) # (Ne, Nq)
    return l

# for linear elasticity
def a_el(Z, Y, coord) -> np.ndarray:
    # grad: (2, 2, Ne, Nq)
    lam_dx = coord.dx + (coord.dx.max() - coord.dx.min()) # (1, Ne, Nq); need change for isoparametric P2 mesh
    a = np.zeros((2, 2, coord.shape[1], coord.shape[2])) # (2,2,Ne,Nq)
    a[0,0] = 2.0 * Z.grad[0,0] * Y.grad[0,0] + 0.5 * Z.grad[0,1] * Y.grad[0,1]
    a[0,1] = 0.5 * Z.grad[0,1] * Y.grad[1,0] + Z.grad[0,0] * Y.grad[1,1]
    a[1,0] = 0.5 * Z.grad[1,0] * Y.grad[0,1] + Z.grad[1,1] * Y.grad[0,0]
    a[1,1] = 0.5 * Z.grad[1,0] * Y.grad[1,0] + 2.0 * Z.grad[1,1] * Y.grad[1,1]
    return a * lam_dx[np.newaxis]    


class PhysicalParameters:
    eta_1: float = 10.0
    beta_1: float = 0.1
    beta_s: float = 0.1
    l_s: float = 0.1
    Ca: float = 0.01
    cosY: float = cos(np.pi*2.0/3)

class SolverParemeters:
    dt: float = 1.0/1024
    Te: float = 1.0/8
    startStep: int = 0
    stride: int = 1
    numChekpoint: int = 0
    vis: bool = True


if __name__ == "__main__":

    mesh = Mesh()
    mesh.load("mesh/two-phase.msh")
    setMeshMapping(mesh)
    mesh.add_constraint(lambda x: np.abs(x[:,0]+1.0) < 1e-14, 
                        lambda x: np.abs(x[:,0]-1.0) < 1e-14, 
                        lambda x: x + np.array((2.0, 0.0)), 
                        tol=1e-11)
    
    i_mesh = mesh.view(Measure(1, (3,)))
    setMeshMapping(i_mesh)

    mixed_fe = (TriP2(mesh, 2, periodic=True),  # U
              TriP1(mesh, 1, periodic=True),    # P1
              TriDG0(mesh, 1, periodic=True),   # P0
              i_mesh.coord_fe,                  # X
              LineP1(i_mesh))                   # K
    
    Y_fe = mesh.coord_fe

    # determine the fixed dof and free dof
    top_dof = np.unique(mixed_fe[0].getCellDof(Measure(1, (7, ))))
    bot_dof = np.unique(mixed_fe[0].getCellDof(Measure(1, (4, 5))))
    u_period_dof = np.nonzero(mixed_fe[0].dof_remap != np.arange(mixed_fe[0].num_dof))[0]
    p1_period_dof = np.nonzero(mixed_fe[1].dof_remap != np.arange(mixed_fe[1].num_dof))[0]
    cl_dof = mixed_fe[3].getCellDof(Measure(0, (9,)))

    # let's hope the first dof in P1, P0 is not a slave
    assert mixed_fe[1].dof_remap[0] == 0
    zero_dof = np.array((0,), dtype=np.uint32)
    assert not hasattr(mixed_fe[2], "dof_remap")

    fixed_dof_list = (
        ((top_dof, u_period_dof), (top_dof, u_period_dof, bot_dof)), 
        (zero_dof, p1_period_dof), 
        (zero_dof, ), 
        (None, (cl_dof, )), 
        None
    )
    free_dof = group_dof(mixed_fe, fixed_dof_list)
        
    Y_fix_dof = np.unique(Y_fe.getCellDof(Measure(1)))
    Y_bot_dof = np.unique(Y_fe.getCellDof(Measure(1, (4,5))))
    Y_int_dof = np.unique(Y_fe.getCellDof(Measure(1, (3,))))

    Y_free_dof = group_dof((Y_fe, ), (((Y_fix_dof, ), (Y_fix_dof ,)), ))

    phys = PhysicalParameters()
    solp = SolverParemeters()
    
    # get the piecewise constant viscosity and slip coefficient
    eta = np.where(mesh.cell[2][:,-1] == 1, phys.eta_1, 1.0)
    bottom_flag = (mesh.cell[1][:,-1] == 5) | (mesh.cell[1][:,-1] == 4)
    bottom_tag = mesh.cell[1][bottom_flag, -1]
    beta = np.where(bottom_tag == 5, phys.beta_1, 1.0)

    # initialize the assembler
    asms = (
        assembler(mixed_fe[0], mixed_fe[0], Measure(2), order=3), 
        assembler(mixed_fe[0], mixed_fe[1], Measure(2), order=3), 
        assembler(mixed_fe[0], mixed_fe[2], Measure(2), order=2), 
        assembler(mixed_fe[0], mixed_fe[4], Measure(1, (3,)), order=4), 
        assembler(mixed_fe[0], mixed_fe[0], Measure(1, (4, 5)), order=5), # [4]: for navier slip
        assembler(mixed_fe[3], mixed_fe[3], Measure(1), order=3), 
        assembler(mixed_fe[3], mixed_fe[4], Measure(1), order=3), 
        assembler(mixed_fe[4], mixed_fe[0], Measure(1, (3,)), order=4),  # [7]: for advection by u
        assembler(mixed_fe[3], mixed_fe[3], Measure(0, (9,)), order=1, geom_hint=("f", )), # [8]: slip at cl
        assembler(Y_fe ,Y_fe, Measure(2), order=3) # [9]: for linear elasticity
    )

    # initialize the unknown
    u = Function(mixed_fe[0])
    p1 = Function(mixed_fe[1])
    p0 = Function(mixed_fe[2])
    x_m = Function(mixed_fe[3])
    x = Function(mixed_fe[3])
    kappa = Function(mixed_fe[4])

    Y = Function(Y_fe)
    Yg = np.zeros_like(Y)

    ax = None
    if solp.vis:
        pyplot.ion()
        fig, ax = pyplot.subplots()
        ax.axis("equal")

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
            pyplot.tripcolor(mesh.coord_map[:,0], mesh.coord_map[:,1], facecolors=mesh.cell[2][:,-1], triangles=mesh.cell[2][:,:-1])
            pyplot.triplot(mesh.coord_map[:,0], mesh.coord_map[:,1], triangles=mesh.cell[2][:,:-1])
            ax.axis("equal")
            pyplot.draw()
            pyplot.pause(0.01)
            # maybe some more efficient ploting

        # first get the interface parametrization
        np.copyto(x_m, i_mesh.coord_map)

        [asm.updateGeometry() for asm in asms]

        # assemble the coupled system
        A_wu = asms[0].bilinear(Form(a_wu, "grad"), eta=eta)
        B_wp1 = asms[1].bilinear(Form(b_wp))
        B_wp0 = asms[2].bilinear(Form(b_wp))
        A_wk = asms[3].bilinear(Form(a_wk, "f"))
        S_wu = asms[4].bilinear(Form(a_slip_wu, "f"), beta=beta)

        A_gx = asms[5].bilinear(Form(a_gx, "grad"))
        A_gk = asms[6].bilinear(Form(a_wk, "f"))
        A_psiu = asms[7].bilinear(Form(a_psiu, "f"))
        S_gx = asms[8].bilinear(Form(a_slip_gx, "f"))

        A = sparse.bmat(((A_wu + 1.0/phys.l_s*S_wu, -B_wp1, -B_wp0, None, -1.0/phys.Ca*A_wk), 
                        (B_wp1.T, None, None, None, None), 
                        (B_wp0.T, None, None, None, None), 
                        (None, None, None, A_gx + phys.beta_s*phys.Ca/solp.dt*S_gx, A_gk), 
                        (-solp.dt*A_psiu, None, None, A_gk.T, None)), 
                        format="csr")
        
        # assemble the RHS
        L_g = asms[8].linear(Form(l_g, "f"))
        L_g_x = asms[8].linear(Form(l_slip_g_x, "f"), x_m=x_m)
        L_psi_x = asms[7].linear(Form(l_psi_x, "f"), x_m=x_m)

        u[:] = 0.0
        p1[:] = 0.0
        p0[:] = 0.0
        L = group_fn(u, p1, p0, phys.cosY*L_g + phys.beta_s*phys.Ca/solp.dt*L_g_x, L_psi_x)

        # Since the essential conditions are all homogeneous,
        # we don't need to homogeneize the system
        sol_vec = np.zeros_like(L)
        
        # solve the linear system
        sol_vec_free = spsolve(A[free_dof][:,free_dof], L[free_dof])
        sol_vec[free_dof] = sol_vec_free
        split_fn(sol_vec, u, p1, p0, x, kappa)
        u.update()
        p1.update()

        # some useful info ...
        i_disp = x - x_m
        # print("Interface displacement = {}".format(np.linalg.norm(i_disp.reshape(-1), np.inf)))

        # solve the displacement on the substrate
        cl_dof = cl_dof.reshape(-1)
        if x[cl_dof[0], 0] > x[cl_dof[1], 0]: # make sure [0] on the left, [1] on the right
            cl_dof = cl_dof[::-1]
        # set up the dirichlet condition
        Yg[:] = 0.0
        Y0_m = mesh.coord_map[Y_bot_dof, 0] # the x coordinate of the grid points on the substrate
        cl_pos = (x_m[cl_dof[0], 0], x_m[cl_dof[1], 0])
        Yg[Y_bot_dof, 0] = np.where(
            Y0_m < cl_pos[0], (Y0_m + 1.0) / (cl_pos[0] + 1.0) * i_disp[cl_dof[0],0], 
            np.where(Y0_m > cl_pos[1], (1.0 - Y0_m) / (1.0 - cl_pos[1]) * i_disp[cl_dof[1],0], 
                     (i_disp[cl_dof[0],0] * (cl_pos[1] - Y0_m) + i_disp[cl_dof[1],0] * (Y0_m - cl_pos[0])) / (cl_pos[1] - cl_pos[0]))
        )
        Yg[Y_int_dof] = i_disp

        # solve the linear elastic equation for the bulk mesh deformation ...
        A_el = asms[9].bilinear(Form(a_el, "grad"))
        L_el = -A_el @ Yg.reshape(-1)
        sol_vec = Yg.copy().reshape(-1)
        sol_vec_free = spsolve(A_el[Y_free_dof][:,Y_free_dof], L_el[Y_free_dof])
        sol_vec[Y_free_dof] = sol_vec_free
        split_fn(sol_vec, Y)

        # move the mesh
        Y += mesh.coord_map
        np.copyto(mesh.coord_map, Y)
        np.copyto(i_mesh.coord_map, x)
    # end time loop
