import numpy as np
from math import cos
from mesh import Mesh, Measure
from mesh_util import splitRefine, setMeshMapping
from fe import TriDG0, TriP1, TriP2, LineP1
from function import Function, split_fn, group_fn
from assemble import assembler, Form
from scipy import sparse
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot

# physical groups from GMSH
group_name = {"fluid_1": 1, "fluid_2": 2, "interface": 3, "dry": 4, "wet": 5, \
             "right": 6, "top": 7, "left": 8, "cl": 9}

def a_wu(w, u, coord, eta) -> np.ndarray:
    # eta: (Ne,)
    # grad: (2, 2, Ne, Nq)
    z = np.zeros((2, 2, coord.shape[1], coord.shape[2]))
    z[0,0] = 2.0 * w.grad[0,0] * u.grad[0,0] + w.grad[0,1] * u.grad[0,1]
    z[1,0] = w.grad[1,0] * u.grad[0,1]
    z[0,1] = w.grad[0,1] * u.grad[1,0]
    z[1,1] = w.grad[1,0] * u.grad[1,0] + 2.0 * w.grad[1,1] * u.grad[1,1]
    return z * eta.reshape(1, 1, -1, 1) * coord.dx[np.newaxis]

def b_wp(w, p, coord) -> np.ndarray:
    # w.grad: (2,2,Ne,Nq)
    # p: (1, 1, Nq)
    z = np.zeros((2, 1, coord.shape[1], coord.shape[2]))
    z[0,0,:,:] = w.grad[0,0,:,:] * p[0,:,:]
    z[1,0,:,:] = w.grad[1,1,:,:] * p[0,:,:]
    return z * coord.dx[np.newaxis]

def a_wk(w, kappa, coord) -> np.ndarray:
    # kappa: (1, 1, Nq)
    # w: (2, 1, Nq)
    # coord.n: (2, Ne, Nq)
    # output: (2, 1, Ne, Nq)
    z = coord.n * w * kappa * coord.dx # (2, Ne, Nq)
    return z[:, np.newaxis]

def a_slip_wu(w, u, coord, beta) -> np.ndarray:
    # u, w: (2, 1, Nq)
    # beta: (Ne, )
    z = np.zeros((2, 2, coord.shape[1], coord.shape[2]))
    z[0,0] = u[0] * w[0] * beta[:,np.newaxis] * coord.dx[0] # (Ne, Nq)
    return z

def a_gx(g, x, coord) -> np.ndarray:
    # grad: (2, 2, Ne, Nq)
    z = np.zeros((2, 2, coord.shape[1], coord.shape[2]))
    z[0,0] = np.sum(x.grad[0] * g.grad[0], axis=0) * coord.dx[0]
    z[1,1] = np.sum(x.grad[1] * g.grad[1], axis=0) * coord.dx[0]
    return z

# def a_gk(g, kappa, coord) -> np.ndarray:
#     return a_wk(g, kappa, coord)

def a_psiu(psi, u, coord) -> np.ndarray:
    # u: (2, 1, Nq)
    # psi: (1, 1, Nq)
    # coord.n: (2, Ne, Nq)
    z = u * coord.n * psi * coord.dx # (2, Ne, Nq)
    return z[np.newaxis, :]

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
    z = np.zeros((2, 2, coord.shape[1], coord.shape[2]))
    z[0,0] = g[0] * x[0]
    return z

def l_slip_g_x(g, coord, x_m) -> np.ndarray:
    # g: (2, 1, Nq)
    # x_m: (2, Ne, Nq)
    z = np.zeros((2, coord.shape[1], coord.shape[2]))
    z[0] = g[0] * x_m[0] # (Ne, Nq)
    return z

def l_g(g, coord, cosY) -> np.ndarray:
    # cosY: scalar
    # coord: (2, Ne, Nq)
    # g: (2, 1, Nq)
    z = np.zeros((2, coord.shape[1], coord.shape[2]))
    z[0] = cosY * np.where(coord[0] > 0, g[0], -g[0]) # (Ne, Nq)
    return z


class PhysicalParameters:
    eta_1: float = 10.0
    beta_1: float = 0.1
    beta_s: float = 0.1
    l_s: float = 0.1
    Ca: float = 0.01
    cosY: float = cos(np.pi*2.0/3)

class SolverParemeters:
    dt: float = 1.0/1024
    Te: float = 1.0/1024
    startStep: int = 0
    stride: int = 1
    numChekpoint: int = 0


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

    phys = PhysicalParameters()
    solp = SolverParemeters()
    
    # get the piecewise constant viscosity and slip coefficient
    eta = np.where(mesh.cell[2][:,-1] == 1, phys.eta_1, 1.0)
    bottom_flag = (mesh.cell[1][:,-1] == group_name["wet"]) | (mesh.cell[1][:,-1] == group_name["dry"])
    bottom_tag = mesh.cell[1][bottom_flag, -1]
    beta = np.where(bottom_tag == group_name["wet"], phys.beta_1, 1.0)

    # initialize the assembler
    asms = (
        assembler(mixed_fe[0], mixed_fe[0], Measure(2), order=3), 
        assembler(mixed_fe[0], mixed_fe[1], Measure(2), order=3), 
        assembler(mixed_fe[0], mixed_fe[2], Measure(2), order=2), 
        assembler(mixed_fe[0], mixed_fe[4], Measure(1, (3,)), order=4), 
        assembler(mixed_fe[0], mixed_fe[0], Measure(1, (4, 5)), order=5), # [4]: for navier slip
        assembler(mixed_fe[3], mixed_fe[3], Measure(1), order=3), 
        assembler(mixed_fe[3], mixed_fe[4], Measure(1), order=3), 
        assembler(mixed_fe[4], mixed_fe[0], Measure(1, (3,)), order=4), 
        assembler(mixed_fe[3], mixed_fe[3], Measure(0, (9,)), order=1, geom_hint=("f", )), # [8]: slip at cl
    )

    # initialize the unknown
    u = Function(mixed_fe[0])
    p1 = Function(mixed_fe[1])
    p0 = Function(mixed_fe[2])
    x_m = Function(mixed_fe[3])
    x = Function(mixed_fe[3])
    kappa = Function(mixed_fe[4])

    Y = Function(mesh.coord_fe) # coordinate map for the bulk mesh

    m = solp.startStep
    while True:
        t = m * solp.dt
        if t >= solp.Te:
            break
        print("t = {0:.4f}".format(t))
        m += 1

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
        L_g = asms[8].linear(Form(l_g, "f"), cosY=phys.cosY)
        L_g_x = asms[8].linear(Form(l_slip_g_x, "f"), x_m=x_m)
        L_psi_x = asms[7].linear(Form(l_psi_x, "f"), x_m=x_m)

        u[:] = 0.0
        p1[:] = 0.0
        p0[:] = 0.0
        L = group_fn(u, p1, p0, L_g + phys.beta_s*phys.Ca/solp.dt*L_g_x, solp.dt*L_psi_x)

        # Since the essential conditions are all homogeneous,
        # we don't need to homogeneize the system
        sol_vec = np.zeros_like(L)

        # determine the fixed dof and free dof
        top_dof = np.unique(mixed_fe[0].getCellDof(Measure(1, (7, ))))
        bot_dof = np.unique(mixed_fe[0].getCellDof(Measure(1, (4, 5))))
        period_dof = np.nonzero(mixed_fe[0].dof_remap != np.arange(mixed_fe[0].num_dof))[0]
        cl_dof = mixed_fe[3].getCellDof(Measure(0, (9,)))

        # let's hope the first dof in P1, P0 is not a slave
        assert mixed_fe[1].dof_remap[0] == 0
        zero_dof = np.array((0,), dtype=np.uint32)

        fixed_dof_list = (
            ((top_dof, period_dof), (top_dof, period_dof, bot_dof)), 
            (zero_dof, ), 
            (zero_dof, ), 
            (None, (cl_dof, )), 
            None
        )

        free_dof = np.ones_like(L, dtype=np.bool8)
        # combine these dof to get the free dof
        base_index = 0
        for fe, dof in zip(mixed_fe, fixed_dof_list):
            if fe.num_copy == 1:
                dof = (dof, )
            for c in range(fe.num_copy):
                if dof[c] is None:
                    continue
                assert isinstance(dof[c], tuple)
                for dd in dof[c]:
                    free_dof[dd.reshape(-1)*fe.num_copy+c+base_index] = False
            base_index += fe.num_dof * fe.num_copy
        assert base_index == L.size
        
        # solve the linear system
        sol_vec_free = spsolve(A[free_dof][:,free_dof], L[free_dof])
        sol_vec[free_dof] = sol_vec_free
        split_fn(sol_vec, u, p1, p0, x, kappa)
        u.update()
        p1.update()

        # some useful info ...
        x_m -= x
        print("Interface displacement = {}".format(np.linalg.norm(x_m.reshape(-1), np.inf)))

        # solve the linear elastic equation for the bulk mesh deformation

        # move the mesh
        # np.copyto(i_mesh.coord_map, x)
        # np.copyto(mesh.coord_map, Y)

    pass
