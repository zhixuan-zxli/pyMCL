from sys import argv
import numpy as np
from fem.mesh import *
from fem.mesh_util import *
from fem.element import *
from fem.funcspace import *
from fem.function import *
from fem.funcbasis import *
from fem.form import *
from fem.post import *
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot

@BilinearForm
def a00(eta, x, z) -> np.ndarray:
    # x.grad (3, 3, Ne, Nq)
    # eta.grad (3, 3, Ne, Nq)
    return np.sum(x.grad * eta.grad, axis=(0,1))[np.newaxis] * z.dx

@BilinearForm
def a10(chi, x, z) -> np.ndarray:
    # x : (3, 1, Nq)
    # z.cn : (3, Ne, Nq)
    # chi : (1, 1, Nq)
    return np.sum(x * z.cn, axis=0, keepdims=True) * chi * z.dx

@BilinearForm
def a11(chi, kappa, z) -> np.ndarray:
    # kappa : (1, 1, Nq)
    # chi : (1, 1, Nq)
    return kappa * chi * z.dx

@LinearForm
def l1(chi, z, x_m) -> np.ndarray:
    # chi : (1,1,Nq)
    # z.n : (3, Ne, Nq)
    # x_m : (3, Ne, Nq)
    return np.sum(x_m * z.cn, axis=0, keepdims=True) * chi * z.dx

class SolverParameters:
    dt: float = 1.0/1000
    maxStep: int = 60

if __name__ == "__main__":

    vis = len(argv) >= 2 and bool(argv[1])

    solp = SolverParameters()

    mesh = Mesh()
    mesh.load("mesh/dumbbell.msh")
    setMeshMapping(mesh)

    mixed_fs = (
        mesh.coord_fe, # X
        FunctionSpace(mesh, TriP1),  # kappa
    )
    dx = Measure(mesh, 2, order=3)
    x_b = FunctionBasis(mixed_fs[0], dx)
    k_b = FunctionBasis(mixed_fs[1], dx)

    x_m = Function(mixed_fs[0])
    x = Function(mixed_fs[0])
    k = Function(mixed_fs[1])


    # visualize
    if vis:
        pyplot.ion()
        fig = pyplot.figure()
        ax = fig.add_subplot(projection="3d")
        nv = NodeVisualizer(mesh, mixed_fs[1])

    def redraw() -> None:
        z = nv.remap(x_m, num_copy=3)
        ax.clear()
        ts = ax.plot_trisurf(z[:,0], z[:,1], z[:,2], triangles=mesh.cell[2])
        ts.set(linewidth=0.5)
        ts.set_edgecolor("tab:blue")
        ts.set_facecolor((0.0, 0.0, 0.0, 0.0))
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_zlim(-0.8, 0.8)
        pyplot.draw()
        pyplot.pause(1e-3)

    for m in range(solp.maxStep):
        print("Solving t = {0:.4f}, ".format((m+1) * solp.dt), end="")
        # get the current mesh
        x_m[:] = mesh.coord_map
        x[:] = 0.0
        if vis:
            redraw()
        # assemble the system
        dx.update()
        x_b.update()
        k_b.update()
        A00 = a00.assemble(x_b, x_b, dx)
        A10 = a10.assemble(k_b, x_b, dx)
        A11 = a11.assemble(k_b, k_b, dx)
        L1 = l1.assemble(k_b, dx, x_m=x_m._interpolate(dx))
        Aa = bmat(((A00, A10.T), (A10, -solp.dt * A11)), format="csr")
        La = group_fn(x, L1)
        # solve the system
        sol_vec = spsolve(Aa, La)
        split_fn(sol_vec, x, k)
        # update the mesh
        mesh.coord_map[:] = x
        disp = x - x_m
        print('disp = {0:.3e}'.format(np.linalg.norm(disp, ord=np.inf)))
        # update plot
        if vis:
            redraw()
    
    print("Finished.\n")
