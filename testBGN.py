from sys import argv
import numpy as np
from fem import *
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
# from scikits.umfpack import spsolve
from matplotlib import pyplot

@BilinearForm
def a00(eta, x, z, _) -> np.ndarray:
    # x.grad (3, 3, Ne, Nq)
    # eta.grad (3, 3, Ne, Nq)
    return np.sum(x.grad * eta.grad, axis=(0,1))[np.newaxis] * z.dx

@BilinearForm
def a10(chi, x, z, _) -> np.ndarray:
    # x : (3, 1, Nq)
    # z.cn : (3, Ne, Nq)
    # chi : (1, 1, Nq)
    return np.sum(x * z.cn, axis=0, keepdims=True) * chi * z.dx

@BilinearForm
def a11(chi, kappa, z, _) -> np.ndarray:
    # kappa : (1, 1, Nq)
    # chi : (1, 1, Nq)
    return kappa * chi * z.dx

class SolverParameters:
    dt: float = 1.0/1000
    maxStep: int = 300

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
    x_basis = FunctionBasis(mixed_fs[0], dx)
    k_basis = FunctionBasis(mixed_fs[1], dx)

    x_m = Function(mixed_fs[0])
    x = Function(mixed_fs[0])
    k = Function(mixed_fs[1])


    # visualize
    if vis:
        pyplot.ion()
        fig = pyplot.figure()
        ax = fig.add_subplot(projection="3d")

    def redraw() -> None:
        ax.clear()
        ts = ax.plot_trisurf(x_m[::3], x_m[1::3], x_m[2::3], triangles=mixed_fs[0].elem_dof[::3].T//3)
        ts.set(linewidth=0.2)
        ts.set_edgecolor("tab:blue")
        ts.set_facecolor((0.0, 0.0, 0.0, 0.0))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        pyplot.draw()
        pyplot.pause(1e-4)

    for m in range(solp.maxStep):
        print("Solving t = {0:.4f}, ".format((m+1) * solp.dt), end="")
        # get the current mesh
        x_m[:] = mesh.coord_map
        x[:] = 0.0
        if vis:
            redraw()
        # assemble the system
        dx.update()
        x_basis.update()
        k_basis.update()
        A00 = a00.assemble(x_basis, x_basis)
        A10 = a10.assemble(k_basis, x_basis)
        A11 = a11.assemble(k_basis, k_basis)
        Aa = bmat(((A00, A10.T), (A10, -solp.dt * A11)), format="csc")
        La = group_fn(x, A10 @ x_m)
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
    pyplot.ioff()
    pyplot.show()
