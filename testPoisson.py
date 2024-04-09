import numpy as np
from fem.mesh import *
from fem.mesh_util import *
from fem.element import *
from fem.funcspace import *
from fem.function import *
from fem.funcbasis import *
from fem.form import *
from fem.util import *
from scipy import sparse
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot

def u_exact(x, y) -> np.ndarray:
    return np.sin(np.pi*x) * np.cos(y)

# def du_exact(x, y) -> np.ndarray:
#     return np.array((np.pi * np.cos(np.pi*x) * np.cos(y), -np.sin(np.pi*x) * np.sin(y)))

@LinearForm
def l(psi, x) -> np.ndarray:
    # psi: (1,1,Nq)
    # coord: (2,Ne,Nq)
    x1, x2 = x
    data = (np.pi**2+1) * np.sin(np.pi*x1) * np.cos(x2) # (Ne, Nq)
    data = data[np.newaxis] * psi * x.dx
    return data

@BilinearForm
def a(v, u, x) -> np.ndarray:
    # grad: (1, 2, Ne, Nq)
    data = np.sum(v.grad * u.grad, axis=1)
    return data * x.dx

# def bc(psi, coord) -> np.ndarray:
#     x, y = coord[0], coord[1] # (Ne, Nq)
#     ngrad = np.sum(du_exact(x, y) * coord.n, axis=0, keepdims=True) # (1, Ne, Nq)
#     return ngrad * psi * coord.dx

# def test(psi, coord) -> np.ndarray:
#     # psi: (1, Ne, Nq)
#     return psi * coord.dx

# def integral(coord, u) -> np.ndarray:
#     return u * coord.dx

@Form
def L2(x, u) -> np.ndarray:
    return u**2 * x.dx


if __name__ == "__main__":

    num_hier = 3
    mesh_table = tuple(f"{i}" for i in range(num_hier))
    error_table = {"infty" : [0.0] * num_hier, "L2" : [0.0] * num_hier}
    mesh = Mesh()
    mesh.load("mesh/unit_square.msh")

    for m in range(num_hier):
        print(f"Testing level {m}...")
        if m > 0:
            mesh = splitRefine(mesh)
        # Affine mesh
        setMeshMapping(mesh)

        fs = FunctionSpace(mesh, TriP2)
        dx = Measure(mesh, 2, order=3)
        u_basis = FunctionBasis(fs, dx)

        A = a.assemble(u_basis, u_basis, dx)
        L = l.assemble(u_basis, dx)

        # impose the boundary condition
        bdof = np.unique(fs.getFacetDof())
        free_dof = group_dof((fs,), (bdof,))

        print("Linear system dimension = {}".format(free_dof.sum()))

        u_err = Function(fs)
        u = Function(fs)
        u_err[:] = u_exact(fs.dof_loc[:,0], fs.dof_loc[:,1])
        u[bdof] = u_err[bdof]

        L = L - A @ u # homogeneize the boundary condition

        u_vec = spsolve(A[free_dof][:,free_dof], L[free_dof])
        u[free_dof] = u_vec
        
        # ax_err = fig.add_subplot(1, 2, 2, projection="3d")
        # ax_err.plot_trisurf(mesh.point[:,0], mesh.point[:,1], u_err.ravel(), triangles=mesh.cell[2][:,:-1], cmap=pyplot.cm.Spectral)
        # pyplot.show()
        u_err = u - u_err
        error_table["infty"][m] = np.linalg.norm(u_err, ord=np.inf)
        error_table["L2"][m] = np.sqrt(L2.assemble(dx, u=u_err._interpolate(dx)))

    printConvergenceTable(mesh_table, error_table)
    