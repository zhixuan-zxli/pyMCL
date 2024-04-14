import numpy as np
from fem import *
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot
from colorama import Fore, Style

def u_exact(x) -> np.ndarray:
    # return x * (1.0-x)
    return np.sin(np.pi*x)**2

@LinearForm
def l(v, x) -> np.ndarray:
    # psi: (1,1,Nq)
    # x: (1,Ne,Nq)
    # return 2.0 * v * x.dx
    return -2.0*np.pi**2*np.cos(2*np.pi*x) * v * x.dx

@BilinearForm
def a(v, u, x) -> np.ndarray:
    # grad: (1, 1, Ne, Nq)
    data = np.sum(v.grad * u.grad, axis=1)
    return data * x.dx

@Functional
def L2(x, u) -> np.ndarray:
    return u**2 * x.dx

gamma = 1e4 # stablization parameter for DG

@BilinearForm
def a_DG(v: QuadData, u: QuadData, xv: QuadData, xu: QuadData) -> np.ndarray:
    return -0.5 * v * np.sum(u.grad[0] * xv.fn, axis=0, keepdims=True) * xv.ds \
    -0.5 * u * np.sum(v.grad[0] * xu.fn, axis=0, keepdims=True) * xu.ds \
    + gamma * u * v * np.sum(xv.fn * xu.fn, axis=0, keepdims=True)

@BilinearForm
def a_nitsche(v: QuadData, u: QuadData, x: QuadData) -> np.ndarray:
    return -v * np.sum(u.grad[0] * x.fn, axis=0, keepdims=True) * x.ds \
    -u * np.sum(v.grad[0] * x.fn, axis=0, keepdims=True) * x.ds \
    + gamma * u * v

@LinearForm
def l_nitsche(v: QuadData, x: QuadData) -> np.ndarray:
    g = u_exact(x[0])[np.newaxis]
    return -g * np.sum(v.grad[0] * x.fn, axis=0, keepdims=True) * x.ds \
    + gamma * g * v


if __name__ == "__main__":

    test_element = LineP2
    num_hier = 3
    mesh_table = tuple(f"{i}" for i in range(num_hier))
    error_tab_D = {"infty" : [0.0] * num_hier, "L2" : [0.0] * num_hier}
    error_tab_DG = {"infty" : [0.0] * num_hier, "L2" : [0.0] * num_hier}
    mesh = Mesh()
    mesh.load("mesh/unit_interval.msh")

    for m in range(num_hier):
        print(f"Testing level {m} ", end="")
        if m > 0:
            mesh = splitRefine(mesh)
        setMeshMapping(mesh)

        # ==================================================
        # 1. test Dirichlet condition
        fs = FunctionSpace(mesh, test_element)
        dx = Measure(mesh, 1, order=3)
        u_basis = FunctionBasis(fs, dx)

        A = a.assemble(u_basis, u_basis, dx)
        L = l.assemble(u_basis, dx)

        # impose the boundary condition
        bdof = np.unique(fs.getFacetDof((3, 4)))
        free_dof = group_dof((fs,), (bdof,))
        # homogeneize the boundary condition
        u_err = Function(fs)
        u = Function(fs)
        u_err[:] = u_exact(fs.dof_loc[:,0])
        u[bdof] = u_err[bdof]
        L = L - A @ u 
        # solve the linear system
        u_vec = spsolve(A[free_dof][:,free_dof], L[free_dof])
        u[free_dof] = u_vec
        # calculate the errors
        u_err = u - u_err
        error_tab_D["infty"][m] = np.linalg.norm(u_err, ord=np.inf)
        error_tab_D["L2"][m] = np.sqrt(L2.assemble(dx, u=u_err._interpolate(dx)))
        print(".", end="")
        # print(".")

        # ==================================================
        # 4. test DG
        fs = FunctionSpace(mesh, LineDG1)
        # dx remains unchanged
        dS = Measure(mesh, 0, order=3, tags=(INTERIOR_FACET_TAG, ), interiorFacet=True)
        ds = Measure(mesh, 0, order=3, tags=(3,4))
        u_basis = FunctionBasis(fs, dx)
        u_i_basis = FunctionBasis(fs, dS)
        u_s_basis = FunctionBasis(fs, ds)

        A = a.assemble(u_basis, u_basis, dx)
        A_DG = a_DG.assemble(u_i_basis, u_i_basis, dS)
        A_n = a_nitsche.assemble(u_s_basis, u_s_basis, ds)
        L = l.assemble(u_basis, dx)
        L_n = l_nitsche.assemble(u_s_basis, ds)

        # solve the linear system
        u = Function(fs)
        # u[:] = spsolve(A + A_n, L + L_n)
        u[:] = spsolve(A + A_DG + A_n, L + L_n)

        # calculate the error
        u_err = Function(fs)
        u_err[:] = u_exact(fs.dof_loc[:,0])
        u_err = u - u_err
        error_tab_DG["infty"][m] = np.linalg.norm(u_err, ord=np.inf)
        error_tab_DG["L2"][m] = np.sqrt(L2.assemble(dx, u=u_err._interpolate(dx)))
        print('.')

    print(Fore.GREEN + "Dirichlet problem: " + Style.RESET_ALL)
    printConvergenceTable(mesh_table, error_tab_D)
    print(Fore.GREEN + "Discontinuous Galerkin: " + Style.RESET_ALL)
    printConvergenceTable(mesh_table, error_tab_DG)
    