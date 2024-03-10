import numpy as np
from mesh import Mesh
from fe import Measure, TriP1, TriP2
from function import Function
from assemble import assembler, Form, setMeshMapping
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot

def exact(x, y) -> np.ndarray:
    return np.sin(np.pi*x) * np.cos(y)

def exact_grad(x, y) -> np.ndarray:
    return np.array((np.pi * np.cos(np.pi*x) * np.cos(y), -np.sin(np.pi*x) * np.sin(y)))

def rhs(psi, coord) -> np.ndarray:
    # psi: (1,1,Nq)
    # coord: (2,Ne,Nq)
    x, y = coord[0], coord[1]
    data = (np.pi**2+1) * np.sin(np.pi*x) * np.cos(y) # (Ne, Nq)
    data = data[np.newaxis] * psi * coord.dx
    return data

def a(phi, psi, coord) -> np.ndarray:
    # grad: (1, 2, Ne, Nq)
    data = np.sum(phi.grad * psi.grad, axis=1)
    data = data * coord.dx
    return data[:,np.newaxis]

def bc(psi, coord) -> np.ndarray:
    x, y = coord[0], coord[1] # (Ne, Nq)
    ngrad = np.sum(exact_grad(x, y) * coord.n, axis=0, keepdims=True) # (1, Ne, Nq)
    return ngrad * psi * coord.dx

def integral(coord, u) -> np.ndarray:
    return u * coord.dx


if __name__ == "__main__":
    mesh = Mesh()
    mesh.load("mesh/unit_square.msh")
    setMeshMapping(mesh)
    u_space = TriP1(mesh)
    asm_2 = assembler(u_space, u_space, Measure(2, None), 3)
    f = asm_2.linear(Form(rhs, "f"))
    A = asm_2.bilinear(Form(a, "grad"))
    asm_1 = assembler(u_space, None, Measure(1, (2,)), 3, ("f", "grad", "dx", "n"))
    g = asm_1.linear(Form(bc, "f"))
    u_sol = spsolve(A, f + g).reshape(-1, 1)
    u_err = Function(u_space)
    u_err[:] = u_sol
    u_err -= asm_2.functional(Form(integral, "f"), u=u_err) / 1.0
    #
    fig = pyplot.figure()
    ax_sol = fig.add_subplot(1, 2, 1, projection="3d")
    ax_sol.plot_trisurf(mesh.point[:,0], mesh.point[:,1], u_err.ravel(), triangles=mesh.cell[2][:,:-1], cmap=pyplot.cm.Spectral)
    #
    u_exact = exact(mesh.point[:, 0], mesh.point[:, 1]).reshape(-1, 1)
    u_err -= u_exact
    u_err = u_err - asm_2.functional(Form(integral, "f"), u=u_err) / 1.0
    #
    ax_err = fig.add_subplot(1, 2, 2, projection="3d")
    ax_err.plot_trisurf(mesh.point[:,0], mesh.point[:,1], u_err.ravel(), triangles=mesh.cell[2][:,:-1], cmap=pyplot.cm.Spectral)
    # pyplot.show()
    print("Max-norm of error = {0:.3e}".format(np.linalg.norm(u_err, ord=np.inf)))
