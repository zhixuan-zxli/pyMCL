import numpy as np
from mesh import Mesh
from mesh_util import splitRefine, setMeshMapping
from fe import Measure, TriP1, TriP2
from function import Function
from assemble import assembler, Form
from scipy import sparse
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot
from util import printConvergenceTable
from math import sqrt

def exact(x, y) -> np.ndarray:
    return np.sin(2.0*np.pi*x) * np.cos(np.pi*y)**2

def exact_grad(x, y) -> np.ndarray:
    return np.array((2.0*np.pi*np.cos(2.0*np.pi*x) * np.cos(np.pi*y)**2, 
                     -np.pi*np.sin(2.0*np.pi*x) * np.sin(2.0*np.pi*y)))

def f(psi, coord) -> np.ndarray:
    # psi: (1,1,Nq)
    # coord: (2,Ne,Nq)
    x, y = coord[0], coord[1]
    data = 2.0*np.pi**2 * np.sin(2.0*np.pi*x) * (1.0 + 2.0 * np.cos(2.0*np.pi*y)) # (Ne, Nq)
    data = data[np.newaxis] * psi * coord.dx
    return data

def a(phi, psi, coord) -> np.ndarray:
    # grad: (1, 2, Ne, Nq)
    data = np.sum(phi.grad * psi.grad, axis=1)
    data = data * coord.dx
    return data[:,np.newaxis]

def g(psi, coord) -> np.ndarray:
    x, y = coord[0], coord[1] # (Ne, Nq)
    ngrad = np.sum(exact_grad(x, y) * coord.n, axis=0, keepdims=True) # (1, Ne, Nq)
    return ngrad * psi * coord.dx

def test(psi, coord) -> np.ndarray:
    # psi: (1, Ne, Nq)
    return psi * coord.dx

def integral(coord, u) -> np.ndarray:
    return u * coord.dx

def L2(coord, u) -> np.ndarray:
    return u**2 * coord.dx


if __name__ == "__main__":

    num_hier = 4
    mesh_table = tuple(f"{i}" for i in range(num_hier))
    error_table = {"infty" : [0.0] * num_hier, "L2" : [0.0] * num_hier}
    mesh = Mesh()

    for m in range(num_hier):
        print(f"Testing level {m}...")
        if m == 0:
            mesh.load("mesh/unit_square.msh")
        else:
            mesh = splitRefine(mesh)
        setMeshMapping(mesh) # affine mesh
        mesh.add_constraint(lambda x: np.abs(x[:,0]) < 1e-14, 
                            lambda x: np.abs(x[:,0]-1.0) < 1e-14, 
                            lambda x: x + np.array(((1.0, 0.0))), 
                            tol=1e-11)

        fe = TriP2(mesh, 1, periodic=True)

        asm_2 = assembler(fe, fe, Measure(2), order=3)
        F = asm_2.linear(Form(f, "f"))
        A = asm_2.bilinear(Form(a, "grad"))
        V = asm_2.linear(Form(test, "f"))
        G = assembler(fe, None, Measure(1, (2,4)), 3).linear(Form(g, "f"))

        # the augmented system
        Aa = sparse.bmat(((A, V), (V.T, None)), format="csr")
        Fa = np.vstack((F+G, 0.0))

        # get freeze dof
        slave_dof = np.nonzero(fe.dof_remap != np.arange(fe.num_dof))[0]
        free_dof = np.ones((fe.num_dof + 1, ), dtype=np.bool8)
        free_dof[slave_dof] = False

        # system the system
        sol_vec = spsolve(Aa[free_dof][:,free_dof], Fa[free_dof])
        u = Function(fe)
        u[free_dof[:-1],0] = sol_vec[:-1]
        u.update()

        u_err = Function(fe)
        u_err[:,0] = exact(fe.dofloc[:,0], fe.dofloc[:,1])
        u_err -= u
        u_err -= asm_2.functional(Form(integral, "f"), u=u_err) / 1.0
        
        # ax_err = fig.add_subplot(1, 2, 2, projection="3d")
        # ax_err.plot_trisurf(mesh.point[:,0], mesh.point[:,1], u_err.ravel(), triangles=mesh.cell[2][:,:-1], cmap=pyplot.cm.Spectral)
        # pyplot.show()
        error_table["infty"][m] = np.linalg.norm(u_err, ord=np.inf)
        error_table["L2"][m] = sqrt(asm_2.functional(Form(L2, "f"), u=u_err))

    printConvergenceTable(mesh_table, error_table)
    