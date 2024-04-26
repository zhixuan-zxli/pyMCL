import numpy as np
from fem import *
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot
from colorama import Fore, Style

def u_exact(x, y) -> np.ndarray:
    return np.array(
        (np.sin(np.pi*x)**2 * np.sin(2*np.pi*y), 
         -np.sin(2*np.pi*x) * np.sin(np.pi*y)**2)
    )

def du_exact(x, y) -> np.ndarray:
    return np.array(
        ((np.pi * np.sin(2.0*np.pi*x) * np.sin(2.0*np.pi*y), 2.0*np.pi * np.sin(np.pi*x)**2 * np.cos(2.0*np.pi*y)), 
         (-2*np.pi * np.cos(2.0*np.pi*x) * np.sin(np.pi*y)**2, -np.pi * np.sin(2.0*np.pi*x) * np.sin(2.0*np.pi*y)))
    )

def p_exact(x, y) -> np.ndarray:
    return np.sin(np.pi*x)**2 * np.cos(np.pi*y)

def dp_exact(x, y) -> np.ndarray:
    return np.array(
        (np.pi * np.sin(2.0*np.pi*x) * np.cos(np.pi*y), 
         -np.pi * np.sin(np.pi*x)**2 * np.sin(np.pi*y))
    )

def f_exact(x, y) -> np.ndarray:
    diffu = np.array(
        (-2.0*np.pi**2 * np.sin(2.0*np.pi*y) * (1.0 + 2.0*np.sin(np.pi*x)) * (1.0 - 2.0*np.sin(np.pi*x)), 
         -2.0*np.pi**2 * np.sin(2.0*np.pi*x) * (2.0*np.sin(np.pi*y) + 1.0) * (2.0*np.sin(np.pi*y) - 1.0))
    )
    return diffu - dp_exact(x, y)

@BilinearForm
def a(v, u, x) -> np.ndarray:
    # grad: (2, 2, Ne, Nq)
    z = np.zeros(x.shape[1:])
    for i, j in (0,0), (0,1), (1,0), (1,1):
        z += (u.grad[i,j] + u.grad[j,i]) * v.grad[i,j]
    return z[np.newaxis] * x.dx

@BilinearForm
def b(v, p, x) -> np.ndarray:
    # v.grad: (2,2,Ne,Nq)
    z = (v.grad[0,0] + v.grad[1,1]) * p[0]
    return z[np.newaxis] * x.dx

@LinearForm
def l(v, x) -> np.ndarray:
    return np.sum(f_exact(x[0], x[1]) * v, axis=0, keepdims=True) * x.dx

@LinearForm
def g(v, x) -> np.ndarray:
    x1, x2 = x[0], x[1]
    du = du_exact(x1, x2) # (2, 2, Ne, Nq)
    Du = du + du.transpose((1,0,2,3)) # 2 times sym grad
    p = p_exact(x1, x2) # (Ne, Nq)
    # x.fn : (2, Ne, Nq)
    Tn = np.sum(Du * x.fn[np.newaxis], axis=1) + p[np.newaxis] * x.fn # (2, Ne, Nq)
    # v : (2, 1, Nq)
    return np.sum(Tn * v, axis=0, keepdims=True) * x.ds

@Functional
def integral_P1P0(x, p1, p0) -> np.ndarray:
    return (p1 + p0) * x.dx

@Functional
def L2_P1P0(x, p1, p0) -> np.ndarray:
    return np.sum((p1+p0)**2, axis=0, keepdims=True) * x.dx

@Functional
def L2(x, u) -> np.ndarray:
    return np.sum(u**2, axis=0, keepdims=True) * x.dx


if __name__ == "__main__":

    num_hier = 3
    mesh_table = tuple(f"{i}" for i in range(num_hier))
    error_head = ("u infty", "u L2", "p L2")
    error_table = {k: [1.0] * num_hier for k in error_head}
    mesh = Mesh()
    mesh.load("mesh/unit_square.msh")
    def periodic_constraint(x: np.ndarray) -> np.ndarray:
        flag = np.abs(x[:,0] - 1.0) < 1e-12
        x[flag, 0] -= 1.0

    for m in range(num_hier):
        print(f"Testing level {m}... ", end="")
        if m > 0:
            mesh = splitRefine(mesh)
        setMeshMapping(mesh)
        # mesh boundary: bottom=2, right=3, top=4, right=5

        U = FunctionSpace(mesh, VectorElement(TriP2, 2), constraint=periodic_constraint)
        P1 = FunctionSpace(mesh, TriP1, constraint=periodic_constraint)
        P0 = FunctionSpace(mesh, TriDG0, constraint=periodic_constraint)

        dx = Measure(mesh, 2, 3)
        ds = Measure(mesh, 1, order=3, tags=(4,))
        u_basis = FunctionBasis(U, dx)
        u_s_basis = FunctionBasis(U, ds)
        p1_basis = FunctionBasis(P1, dx)
        p0_basis = FunctionBasis(P0, dx)

        # assemble the form
        A = a.assemble(u_basis, u_basis, dx)
        B1 = b.assemble(u_basis, p1_basis, dx)
        B0 = b.assemble(u_basis, p0_basis, dx)
        L = l.assemble(u_basis, dx)
        G = g.assemble(u_s_basis, ds)

        # assemble the saddle point system
        u = Function(U)
        p1 = Function(P1)
        p0 = Function(P0)
        # Aa = bmat(((A, B1), (B1.T, None)), format="csr")
        Aa = bmat(((A, B1, B0), (B1.T, None, None), (B0.T, None, None)), format="csr")
        # La = group_fn(L, p1)
        La = group_fn(L+G, p1, p0)

        # impose the Dirichlet condition
        bdof = np.unique(U.getFacetDof((2,)))
        # fdof = group_dof((U, P1), (bdof, np.array((0,))))
        fdof = group_dof((U, P1, P0), (bdof, np.array((0,)), np.array((0,))))
        u_err = Function(U)
        u_err[:] = u_exact(U.dof_loc[::2,0], U.dof_loc[::2,1]).T.reshape(-1)
        p1[0] = p_exact(P1.dof_loc[0,0], P1.dof_loc[0,1]) # need to fix this pressure dof
        
        print("dimension = {}".format(fdof.sum()))

        # Homogeneize and then solve the system
        # sol_vec = group_fn(u, p1)
        sol_vec = group_fn(u, p1, p0)
        La = La - Aa @ sol_vec
        z = spsolve(Aa[fdof][:,fdof], La[fdof])
        sol_vec[fdof] = z
        # split_fn(sol_vec, u, p1)
        split_fn(sol_vec, u, p1, p0)

        # calculate the error
        u_err = u - u_err
        error_table["u infty"][m] = np.linalg.norm(u_err, ord=np.inf)
        error_table["u L2"][m] = np.sqrt(L2.assemble(dx, u=u_err._interpolate(dx)))

        p_err = Function(P1)
        p_err[:] = p_exact(P1.dof_loc[:,0], P1.dof_loc[:,1])
        p_err = p1 - p_err
        p0 -= integral_P1P0.assemble(dx, p1=p_err._interpolate(dx), p0=p0._interpolate(dx)) / 1.0

        # # error_table["p infty"][m] = np.linalg.norm(p_err, ord=np.inf)
        # error_table["p L2"][m] = np.sqrt(L2_P1P0.assemble(dx, p1=p_err._interpolate(dx), p0=0.0))
        error_table["p L2"][m] = np.sqrt(L2_P1P0.assemble(dx, p1=p_err._interpolate(dx), p0=p0._interpolate(dx)))

    print(Fore.GREEN + "\nConvergence: " + Style.RESET_ALL)
    printConvergenceTable(mesh_table, error_table)
    