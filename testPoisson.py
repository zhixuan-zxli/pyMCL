import numpy as np
from fem import *
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot
from colorama import Fore, Style

def u_exact(x, y) -> np.ndarray:
    return np.sin(np.pi*x)**2 * np.cos(y)

def du_exact(x, y) -> np.ndarray:
    return np.array((np.pi * np.sin(2.0*np.pi*x) * np.cos(y), -np.sin(np.pi*x)**2 * np.sin(y)))

@LinearForm
def l(psi, x) -> np.ndarray:
    # psi: (1,1,Nq)
    # coord: (2,Ne,Nq)
    x1, x2 = x
    data = (-2.0 * np.pi**2 * np.cos(2.0*np.pi*x1) + np.sin(np.pi*x1)**2) * np.cos(x2)
    # data = (np.pi**2+1) * np.sin(np.pi*x1) * np.cos(x2) # (Ne, Nq)
    data = data[np.newaxis] * psi * x.dx
    return data

@BilinearForm
def a(v, u, x) -> np.ndarray:
    # grad: (1, 2, Ne, Nq)
    data = np.sum(v.grad * u.grad, axis=1)
    return data * x.dx

@LinearForm
def g(v, x) -> np.ndarray:
    x1, x2 = x[0], x[1] # (Ne, Nq)
    ngrad = np.sum(du_exact(x1, x2) * x.fn, axis=0, keepdims=True) # (1, Ne, Nq)
    return ngrad * v * x.ds

@LinearForm
def test(v, x) -> np.ndarray:
    # v: (1, Ne, Nq)
    return v * x.dx

@Functional
def integral(x, u) -> np.ndarray:
    return u * x.dx

@Functional
def L2(x, u) -> np.ndarray:
    return u**2 * x.dx

gamma = 20.0 # stablization parameter for DG

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
    x1, x2 = x
    g = u_exact(x1, x2)[np.newaxis]
    return -g * np.sum(v.grad[0] * x.fn, axis=0, keepdims=True) * x.ds \
    + gamma * g * v


if __name__ == "__main__":

    test_element = TriP1
    num_hier = 3
    mesh_table = tuple(f"{i}" for i in range(num_hier))
    error_tab_D = {"infty" : [0.0] * num_hier, "L2" : [0.0] * num_hier}
    error_tab_N = {"infty" : [0.0] * num_hier, "L2" : [0.0] * num_hier}
    error_tab_P = {"infty" : [0.0] * num_hier, "L2" : [0.0] * num_hier}
    error_tab_DG = {"infty" : [0.0] * num_hier, "L2" : [0.0] * num_hier}
    mesh = Mesh()
    mesh.load("mesh/unit_square.msh")

    # for enforcing periodic constraint
    def constraint(x: np.ndarray) -> np.ndarray:
        flag = np.abs(x[:,0] - 1.0) < 1e-12
        x[flag,0] -= 1.0

    for m in range(num_hier):
        print(f"Testing level {m} ", end="")
        if m > 0:
            mesh = splitRefine(mesh)
        setMeshMapping(mesh)

        # ==================================================
        # 1. test Dirichlet condition
        fs = FunctionSpace(mesh, test_element)
        dx = Measure(mesh, 2, order=3)
        u_basis = FunctionBasis(fs, dx)

        A = a.assemble(u_basis, u_basis, dx)
        L = l.assemble(u_basis, dx)

        # impose the boundary condition
        bdof = np.unique(fs.getFacetDof((2, 3, 4, 5)))
        free_dof = group_dof((fs,), (bdof,))
        # homogeneize the boundary condition
        u_err = Function(fs)
        u = Function(fs)
        u_err[:] = u_exact(fs.dof_loc[:,0], fs.dof_loc[:,1])
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

        # ==================================================
        # 2. test pure Neumann condition
        ds = Measure(mesh, 1, order=3, tags=(2, 3, 4, 5))
        u_s_basis = FunctionBasis(fs, ds)
        L = l.assemble(u_basis, dx)
        G = g.assemble(u_s_basis, ds)
        V = test.assemble(u_basis, dx)
        # assemble the augmented system
        Aa = bmat(((A, V[:,np.newaxis]), (V[np.newaxis,:], None)), format="csr")
        z = np.zeros((1, ))
        La = group_fn(L+G, z)
        # solve the linear system
        u_vec = spsolve(Aa, La)
        u[:] = u_vec[:-1]
        # calculate the errors
        u_err[:] = u_exact(fs.dof_loc[:,0], fs.dof_loc[:,1])
        u_err = u - u_err
        u_err -= integral.assemble(dx, u=u_err._interpolate(dx)) / 1.0
        error_tab_N["infty"][m] = np.linalg.norm(u_err, ord=np.inf)
        error_tab_N["L2"][m] = np.sqrt(L2.assemble(dx, u=u_err._interpolate(dx)))
        print(".", end="")
        
        # ==================================================
        # 3. test periodic condition
        fs = FunctionSpace(mesh, test_element, constraint=constraint)
        # dx remains unchanged
        u_basis = FunctionBasis(fs, dx)

        A = a.assemble(u_basis, u_basis, dx)
        L = l.assemble(u_basis, dx)

        # impose Dirichlet condition
        bdof = np.unique(fs.getFacetDof((2, 4)))
        free_dof = group_dof((fs,), (bdof,))
        # homogeneize the boundary condition
        u_err = Function(fs)
        u = Function(fs)
        u_err[:] = u_exact(fs.dof_loc[:,0], fs.dof_loc[:,1])
        u[bdof] = u_err[bdof]
        L = L - A @ u 
        # solve the linear system
        u_vec = spsolve(A[free_dof][:,free_dof], L[free_dof])
        u[free_dof] = u_vec
        # calculate the errors
        u_err = u - u_err
        error_tab_P["infty"][m] = np.linalg.norm(u_err, ord=np.inf)
        error_tab_P["L2"][m] = np.sqrt(L2.assemble(dx, u=u_err._interpolate(dx)))
        print(".", end="")

        # ==================================================
        # 4. test DG
        # fs = FunctionSpace(mesh, TriDG1)
        fs = FunctionSpace(mesh, test_element)
        # dx remains unchanged
        # dS = Measure(mesh, 1, order=3, tags=(INTERIOR_FACET_TAG, ), interiorFacet=True)
        # ds remains unchanged
        u_basis = FunctionBasis(fs, dx)
        # u_i_basis = FunctionBasis(fs, dS)
        u_s_basis = FunctionBasis(fs, ds)

        A = a.assemble(u_basis, u_basis, dx)
        # A_DG = a_DG.assemble(u_i_basis, u_i_basis, dS)
        A_n = a_nitsche.assemble(u_s_basis, u_s_basis, ds)
        L = l.assemble(u_basis, dx)
        L_n = l_nitsche.assemble(u_s_basis, ds)

        # solve the linear system
        u = Function(fs)
        u[:] = spsolve(A + A_n, L + L_n)
        # u[:] = spsolve(A + A_DG + A_n, L + L_n)

        # calculate the error
        u_err = Function(fs)
        u_err[:] = u_exact(fs.dof_loc[:,0], fs.dof_loc[:,1])
        u_err = u - u_err
        error_tab_DG["infty"][m] = np.linalg.norm(u_err, ord=np.inf)
        error_tab_DG["L2"][m] = np.sqrt(L2.assemble(dx, u=u_err._interpolate(dx)))
        print('.')

    print(Fore.GREEN + "Dirichlet problem: " + Style.RESET_ALL)
    printConvergenceTable(mesh_table, error_tab_D)
    print(Fore.GREEN + "\nNeumann problem: " + Style.RESET_ALL)
    printConvergenceTable(mesh_table, error_tab_N)
    print(Fore.GREEN + "\nPeriodic problem: " + Style.RESET_ALL)
    printConvergenceTable(mesh_table, error_tab_P)
    print(Fore.GREEN + "Discontinuous Galerkin: " + Style.RESET_ALL)
    printConvergenceTable(mesh_table, error_tab_DG)
    