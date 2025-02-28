from sys import argv
import numpy as np
from dataclasses import dataclass
from fem import *
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot

@dataclass
class SolverParameters:
    dt: float = 1.0/1000
    maxStep: int = 300
    vis: bool = False

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

def testMeanCurvatureFlow(solp: SolverParameters) -> None:
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
    if solp.vis:
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
        print('disp = {0:.3e}'.format(np.linalg.norm(disp, ord=np.inf)/solp.dt))
        # update plot
        if solp.vis:
            redraw()
    
    print("Finished.\n")

@BilinearForm
def b01(x, k, z, _) -> np.ndarray:
    # x: (2, Ne, Nq)
    # k: (2, Ne, Nq)
    return np.sum(x * k, axis=0, keepdims=True) * z.dx

@BilinearForm
def b11(phi, k, z, _) -> np.ndarray:
    # z.cn: (2, Ne, Nq)
    # k.grad: (2, 2, Ne, Nq)
    tau = np.array((-z.cn[1], z.cn[0])) # (2, Ne, Nq)
    r1 = np.sum(k.grad * phi.grad, axis=(0,1)) # (Ne, Nq)
    # r3 = np.sum(np.sum(k.grad * tau[:,np.newaxis], axis=0) * np.sum(phi.grad * tau[:,np.newaxis], axis=0), axis=0) # (Ne, Nq)
    r2 = (k.grad[0,0] + k.grad[1,1]) * (phi.grad[0,0] + phi.grad[1,1]) # (Ne, Nq)
    return (r1 - 1.5*r2)[np.newaxis] * z.dx

def testWillmoreFlow(solp: SolverParameters) -> None:
    mesh = Mesh()
    mesh.load("mesh/modulated_circle.msh")
    setMeshMapping(mesh, order=2)

    sp = mesh.coord_fe # type: FunctionSpace
    dx = Measure(mesh, 1, order=3)
    basis = FunctionBasis(sp, dx)

    x_m = Function(sp)
    x = Function(sp)
    kappa = Function(sp)

    if solp.vis:
        pyplot.ion()
        fig = pyplot.figure()
        ax = fig.add_subplot()
        ax.axis("equal")
        seg_reindex = sp.elem_dof[[0, 4, 2]].T.reshape(-1) // 2

    for m in range(solp.maxStep):
        print("Solving t = {0:.4f}, ".format((m+1) * solp.dt), end="")
        # get the current mesh
        x_m[:] = mesh.coord_map
        x[:] = 0.0
        # assemble the system
        dx.update()
        basis.update()
        B00 = a00.assemble(basis, basis)
        B01 = b01.assemble(basis, basis)
        B11 = b11.assemble(basis, basis)
        Aa = bmat(((B00, B01), (B01, -solp.dt * B11)), format="csc")
        La = group_fn(x, B01 @ x_m)
        # solve the system
        sol_vec = spsolve(Aa, La)
        split_fn(sol_vec, x, kappa)
        # update the mesh
        mesh.coord_map[:] = x
        disp = x - x_m
        print('disp = {:.3e}'.format(np.linalg.norm(disp, ord=np.inf)/solp.dt))
        # update the plot
        if solp.vis:
            ax.clear()
            xx = x.reshape(-1, 2)
            ax.plot(xx[seg_reindex,0], xx[seg_reindex,1], 'bo-', mfc='none')
            ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
            pyplot.draw()
            pyplot.pause(1e-1)
        pass
    print("Finished.\n")


if __name__ == "__main__":

    solp = SolverParameters()
    solp.vis = (len(argv) >= 2) and bool(argv[1])

    solp.dt = 1.0/1000
    solp.maxStep = 300
    testMeanCurvatureFlow(solp)
    solp.dt = 1.0/(1024 * 64)
    solp.maxStep = 128
    testWillmoreFlow(solp)

    pyplot.ioff()
    pyplot.show()
