from sys import argv
import numpy as np
from mesh import Mesh
from mesh_util import setMeshMapping
from fe import Measure, TriP1
from function import Function, group_fn, split_fn
from assemble import assembler, Form
from scipy import sparse
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot

def a10(x, chi, coord) -> np.ndarray:
    # x : (3, 1, Nq)
    # coord.n : (3, Ne, Nq)
    # chi : (1, 1, Nq)
    # return (1, 3, Ne, Nq)
    data = x * coord.n * chi * coord.dx # (3, Ne, Nq)
    return data[np.newaxis]

def a11(kappa, chi, coord) -> np.ndarray:
    # kappa : (1, 1, Nq)
    # chi : (1, 1, Nq)
    return -kappa * chi * coord.dx

def a00(x, eta, coord) -> np.ndarray:
    # x.grad (3, 3, Ne, Nq)
    # eta.grad (3, 3, Ne, Nq)
    data = np.zeros((3, 3, coord.shape[1], coord.shape[2]))
    data[0,0,:,:] = np.sum(x.grad[0,:,:,:] * eta.grad[0,:,:,:], axis=0)
    data[1,1,:,:] = np.sum(x.grad[1,:,:,:] * eta.grad[1,:,:,:], axis=0)
    data[2,2,:,:] = np.sum(x.grad[2,:,:,:] * eta.grad[2,:,:,:], axis=0)
    return data * coord.dx[np.newaxis]

def a01(kappa, eta, coord) -> np.ndarray:
    # kappa : (1, 1, Nq)
    # eta : (3, 1, Nq)
    # coord.n : (3, Ne, Nq)
    # return (3, 1, Ne, Nq)
    data = eta * coord.n * kappa * coord.dx # (3, Ne, Nq)
    return data[:, np.newaxis]

def l1(chi, coord, x_m) -> np.ndarray:
    # chi : (1,1,Nq)
    # coord.n : (3, Ne, Nq)
    # x_m : (3, Ne, Nq)
    return np.sum(x_m * coord.n, axis=0, keepdims=True) * chi * coord.dx

if __name__ == "__main__":

    plot_or_not = len(argv) >= 2 and argv[1] == "yes"

    mesh = Mesh()
    mesh.load("mesh/dumbbell.msh")
    setMeshMapping(mesh)

    X_space = mesh.coord_fe
    K_space = TriP1(mesh, num_copy=1)
    x_m = Function(X_space)
    x = Function(X_space)
    k = Function(K_space)

    params = {"dt" : 1.0/1000, "maxStep" : 60}
    geom_hint = ("f", "grad", "dx", "inv_grad", "n")
    
    # visualize
    if plot_or_not:
        pyplot.ion()
        fig = pyplot.figure()
        ax = fig.add_subplot(projection="3d")
        ts = ax.plot_trisurf(mesh.point[:,0], mesh.point[:,1], mesh.point[:,2], triangles=mesh.cell[2][:, :-1])
        ts.set(linewidth=0.5)
        ts.set_edgecolor("tab:blue")
        ts.set_facecolor((0.0, 0.0, 0.0, 0.0))
        fig.canvas.draw() 
        fig.canvas.flush_events()

    for m in range(params["maxStep"]):
        print("Solving t = {0:.4f}, ".format((m+1) * params["dt"]), end="")
        # get the current mesh
        x_m[:] = mesh.coord_map
        x[:] = 0.0
        # assemble the system
        A00 = assembler(X_space, X_space, Measure(2), 3, geom_hint).bilinear(Form(a00, "grad"))
        A01 = assembler(X_space, K_space, Measure(2), 3, geom_hint).bilinear(Form(a01, "f"))
        A10 = assembler(K_space, X_space, Measure(2), 3, geom_hint).bilinear(Form(a10, "f"))
        A11 = assembler(K_space, K_space, Measure(2), 3, geom_hint).bilinear(Form(a11, "f"))
        L1 = assembler(K_space, None, Measure(2), 3, geom_hint).linear(Form(l1, "f"), x_m = x_m)
        Aa = sparse.bmat(((A00, A01), (A10 / params["dt"], A11)), format="csr")
        La = group_fn(x, L1 / params["dt"])
        # solve the system
        sol_vec = spsolve(Aa, La)
        split_fn(sol_vec, x, k)
        # update the mesh
        mesh.coord_map[:] = x
        disp = (x - x_m).reshape(-1)
        print('disp = {0:.3e}'.format(np.linalg.norm(disp, ord=np.inf)))
        # update plot
        if plot_or_not:
            ax.clear()
            ts = ax.plot_trisurf(x[:,0], x[:,1], x[:,2], triangles=mesh.cell[2][:, :-1])
            ts.set(linewidth=0.5)
            ts.set_edgecolor("tab:blue")
            ts.set_facecolor((0.0, 0.0, 0.0, 0.0))
            fig.canvas.draw() 
            fig.canvas.flush_events()
