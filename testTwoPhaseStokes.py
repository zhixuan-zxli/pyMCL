import numpy as np
from mesh import Mesh, Measure
from mesh_util import splitRefine, setMeshMapping
from fe import TriDG0, TriP1, TriP2
from function import Function, split_fn, group_fn
from assemble import assembler, Form
from scipy import sparse
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot

# physical groups
phy_group = {"fluid_1": 1, "fluid_2": 2, "interface": 3, "dry": 4, "wet": 5, \
             "right": 6, "top": 7, "left": 8, "cl": 9}

def a_uw(u, w, coord, eta) -> np.ndarray:
    # eta: (Ne,)
    # grad: (2, 2, Ne, Nq)
    z = np.zeros_like((2, 2, coord.shape[1], coord.shape[2]))
    z[0,0] = 2.0 * w.grad[0,0] * u.grad[0,0] + w.grad[0,1] * u.grad[0,1]
    z[1,0] = w.grad[1,0] * u.grad[0,1]
    z[0,1] = w.grad[0,1] * u.grad[1,0]
    z[1,1] = w.grad[1,0] * u.grad[1,0] + 2.0 * w.grad[1,1] * u.grad[1,1]
    return z * eta.reshape(1, 1, -1, 1) * coord.dx[np.newaxis]

def b_pw(p, w, coord) -> np.ndarray:
    # w.grad: (2,2,Ne,Nq)
    # p: (1, 1, Nq)
    z = np.zeros((2, 1, coord.shape[1], coord.shape[2]))
    z[0,0,:,:] = w.grad[0,0,:,:] * p[0,:,:]
    z[1,0,:,:] = w.grad[1,1,:,:] * p[0,:,:]
    return z * coord.dx[np.newaxis]

def a_kw(kappa, w, coord) -> np.ndarray:
    # kappa: (1, 1, Nq)
    # w: (2, 1, Nq)
    # coord.n: (2, Ne, Nq)
    # output: (2, 1, Ne, Nq)
    z = coord.n * w * kappa * coord.dx # (2, Ne, Nq)
    return z[:, np.newaxis]

def g_slip(u, w, coord, beta) -> np.ndarray:
    # u, w: (2, 1, Nq)
    # beta: (Ne, )
    z = np.zeros((2, 2, coord.shape[1], coord.shape[2]))
    z[0,0] = u[0] * w[0] * beta[:,np.newaxis] * coord.dx[0] # (Ne, Nq)
    return z

def a_xg(x, g, coord) -> np.ndarray:
    # grad: (2, 2, Ne, Nq)
    z = np.zeros((2, 2, coord.shape[1], coord.shape[2]))
    z[0,0] = np.sum(x.grad[0] * g.grad[0], axis=0) * coord.dx[0]
    z[1,1] = np.sum(x.grad[1] * g.grad[1], axis=0) * coord.dx[0]
    return z

# def a_kg(kappa, g, coord) -> np.ndarray:
#     return a_kw(kappa, g, coord)

def a_xpsi(x, psi, coord) -> np.ndarray:
    # x: (2, 1, Nq)
    # psi: (1, 1, Nq)
    # coord.n: (2, Ne, Nq)
    z = x * coord.n * psi * coord.dx # (2, Ne, Nq)
    return z[np.newaxis, :]

# def a_upsi(u, psi, coord) -> np.ndarray:
#     return a_xpsi(u, psi, coord)

def l_xpsi(psi, coord, x_m) -> np.ndarray:
    # x_m: (2, Ne, Nq)
    # coord.n: (2, Ne, Nq)
    # psi: (1, 1, Nq)
    return np.sum(x_m * coord.n, axis=0, keepdims=True) * psi * coord.dx

if __name__ == "__main__":

    mesh = Mesh()
    mesh.load("mesh/two-phase.msh")
    pyplot.figure()
    mesh.draw()

    interface_mesh = mesh.view(Measure(1, (3, )))
    pyplot.figure()
    interface_mesh.draw()
    pyplot.show()
