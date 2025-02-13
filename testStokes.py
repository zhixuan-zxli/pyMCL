import numpy as np
from fem import *
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
# from matplotlib import pyplot
from colorama import Fore, Style

# =======================================================================
# below lists the exact solution
def u_exact(x, y) -> np.ndarray:
    return np.array(
        (np.sin(x)**2 * np.sin(2*y), 
         -np.sin(2*x) * np.sin(y)**2)
    )

def du_exact(x, y) -> np.ndarray:
    return np.array(
        ((np.sin(2*x) * np.sin(2*y), 2 * np.sin(x)**2 * np.cos(2*y)), 
         (-2 * np.cos(2*x) * np.sin(y)**2, -np.sin(2*x) * np.sin(2*y)))
    )

def p_exact(x, y) -> np.ndarray:
    return np.sin(np.pi*x)**2 * np.cos(np.pi*y)

def dp_exact(x, y) -> np.ndarray:
    return np.array(
        (np.pi * np.sin(2.0*np.pi*x) * np.cos(np.pi*y), 
         -np.pi * np.sin(np.pi*x)**2 * np.sin(np.pi*y))
    )

def f_exact(x, y) -> np.ndarray:
    # this is the Laplacian of u_exact
    diffu = np.array(
        (2 * np.sin(2*y) * (1.0 + 2.0*np.sin(x)) * (1.0 - 2.0*np.sin(x)), 
         2 * np.sin(2*x) * (2.0*np.sin(y) + 1.0) * (2.0*np.sin(y) - 1.0))
    )
    return diffu - dp_exact(x, y)

# =======================================================================
# below defines the forms
@BilinearForm
def a(v, u, x, _) -> np.ndarray:
    # grad: (2, 2, Ne, Nq)
    z = np.zeros(x.shape[1:])
    for i, j in (0,0), (0,1), (1,0), (1,1):
        z += (u.grad[i,j] + u.grad[j,i]) * v.grad[i,j]
    return z[np.newaxis] * x.dx

@BilinearForm
def b(v, p, x, _) -> np.ndarray:
    # v.grad: (2,2,Ne,Nq)
    z = (v.grad[0,0] + v.grad[1,1]) * p[0]
    return z[np.newaxis] * x.dx

@LinearForm
def l(v, x) -> np.ndarray:
    return np.sum(f_exact(x[0], x[1]) * v, axis=0, keepdims=True) * x.dx

# to impose the normal velocity weakly
@BilinearForm
def stress_nor(v, u, x, _) -> np.ndarray:
    Du = u.grad + u.grad.transpose((1,0,2,3)) # shape (2, 2, Ne, Nq)
    n_Tu_n = np.sum(np.sum(Du * x.fn[np.newaxis], axis=1) * x.fn, axis=0) # shape (Ne, Nq)
    u_n = np.sum(u * x.fn, axis=0) # shape (Ne, Nq)
    Dv = v.grad + v.grad.transpose((1,0,2,3)) # shape (2, 2, Ne, Nq)
    n_Tv_n = np.sum(np.sum(Dv * x.fn[np.newaxis], axis=1) * x.fn, axis=0) # shape (Ne, Nq)
    v_n = np.sum(v * x.fn, axis=0) # shape (Ne, Nq)
    return (-n_Tu_n * v_n + n_Tv_n * u_n)[np.newaxis] * x.ds

@BilinearForm
def stress_nor_p(v, p, x, _) -> np.ndarray:
    # p : (1, 1, Nq)
    v_n = np.sum(v * x.fn, axis=0) # shape (Ne, Nq)
    return p * v_n[np.newaxis] * x.ds

@LinearForm
def stress_nor_rhs(v, x) -> np.ndarray:
    Dv = v.grad + v.grad.transpose((1,0,2,3)) # shape (2, 2, Ne, Nq)
    n_Tv_n = np.sum(np.sum(Dv * x.fn[np.newaxis], axis=1) * x.fn, axis=0) # shape (Ne, Nq)
    g = np.sum(u_exact(x[0], x[1]) * x.fn, axis=0) # shape (Ne, Nq)
    return (n_Tv_n * g)[np.newaxis] * x.ds

@LinearForm
def stress_nor_p_rhs(p, x) -> np.ndarray:
    g = np.sum(u_exact(x[0], x[1]) * x.fn, axis=0) # shape (Ne, Nq)
    return -p * g[np.newaxis] * x.ds

@BilinearForm
def stabilization(v, u, x, _) -> np.ndarray:
    u_n = np.sum(u * x.fn, axis=0) # shape (Ne, Nq)
    v_n = np.sum(v * x.fn, axis=0) # shape (Ne, Nq)
    return (u_n * v_n)[np.newaxis]

@LinearForm
def stabilization_rhs(v, x) -> np.ndarray:
    v_n = np.sum(v * x.fn, axis=0) # shape (Ne, Nq)
    g = np.sum(u_exact(x[0], x[1]) * x.fn, axis=0)
    return (v_n * g)[np.newaxis]

# handle the tangential stress at the bottom and top boundary
@LinearForm
def stress_tan(v, x) -> np.ndarray:
    x1, x2 = x[0], x[1]
    du = du_exact(x1, x2) # (2, 2, Ne, Nq)
    Du = du + du.transpose((1,0,2,3)) # 2 times sym grad
    # x.fn : (2, Ne, Nq)
    Tn = np.sum(Du * x.fn[np.newaxis], axis=1) # (2, Ne, Nq)
    # v : (2, 1, Nq)
    return (Tn[0] * v[0])[np.newaxis] * x.ds

@Functional
def integral_P1(x, p1) -> np.ndarray:
    return p1 * x.dx

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

    gamma = 10.0

    num_hier = 3
    mesh_table = tuple(f"{i}" for i in range(num_hier))
    error_head = ("u infty", "u L2", "p L2")
    strong_error_table = {k: [1.0] * num_hier for k in error_head}
    weak_error_table = {k: [1.0] * num_hier for k in error_head}
    mesh = Mesh()
    mesh.load("mesh/unit_square.msh")
    # def periodic_constraint(x: np.ndarray) -> np.ndarray:
    #     flag = np.abs(x[:,0] - 1.0) < 1e-12
    #     x[flag, 0] -= 1.0

    for m in range(num_hier):
        print(f"Testing level {m}... ", end="")
        if m > 0:
            mesh = splitRefine(mesh)
        setMeshMapping(mesh)
        # mesh boundary: bottom=2, right=3, top=4, left=5

        U = FunctionSpace(mesh, VectorElement(TriP2, num_copy=2))
        P1 = FunctionSpace(mesh, TriP1)
        P0 = FunctionSpace(mesh, TriDG0)

        dx = Measure(mesh, 2, order=3)
        ds = Measure(mesh, 1, order=3, tags=(2, 4))
        u_basis = FunctionBasis(U, dx)
        u_s_basis = FunctionBasis(U, ds)
        p1_basis = FunctionBasis(P1, dx)
        p1_s_basis = FunctionBasis(P1, ds)
        p0_basis = FunctionBasis(P0, dx)
        p0_s_basis = FunctionBasis(P0, ds)

        # =======================================================================
        # Imposing the normal velocity using Nitsche method
        # assemble the form
        A = a.assemble(u_basis, u_basis)
        B1 = b.assemble(u_basis, p1_basis)
        B0 = b.assemble(u_basis, p0_basis)
        F_TAU = stress_tan.assemble(u_s_basis)
        A_nit = stress_nor.assemble(u_s_basis, u_s_basis)
        A_nit_p = stress_nor_p.assemble(u_s_basis, p1_s_basis)
        A_nit_p0 = stress_nor_p.assemble(u_s_basis, p0_s_basis)
        STAB = stabilization.assemble(u_s_basis, u_s_basis)
        L = l.assemble(u_basis)
        L_nit = stress_nor_rhs.assemble(u_s_basis)
        L_nit_p = stress_nor_p_rhs.assemble(p1_s_basis)
        L_nit_p0 = stress_nor_p_rhs.assemble(p0_s_basis)
        L_STAB = stabilization_rhs.assemble(u_s_basis)

        # assemble the saddle point system
        u = Function(U)
        p1 = Function(P1)
        p0 = Function(P0)
        # Aa = bmat(((A + A_nit + gamma*STAB, -B1+A_nit_p), (B1.T-A_nit_p.T, None)), format="csc")
        Aa = bmat(((A + A_nit + gamma*STAB, -B1+A_nit_p, -B0+A_nit_p0), (B1.T-A_nit_p.T, None, None), (B0.T-A_nit_p0.T, None, None)), format="csc")
        # La = group_fn(-L + L_nit + gamma * L_STAB + F_TAU, p1 + L_nit_p)
        La = group_fn(-L + L_nit + gamma * L_STAB + F_TAU, p1 + L_nit_p, p0 + L_nit_p0)

        # impose the Dirichlet condition
        bdof = np.where((U.dof_loc[:,0] < 1e-12) | (U.dof_loc[:,0] > 1-1e-12))[0]
        # fdof = group_dof((U, P1), (bdof, np.array((0,))))
        fdof = group_dof((U, P1, P0), (bdof, np.array((0,)), np.array((0,))))
        u_err = Function(U)
        u_err[:] = u_exact(U.dof_loc[::2,0], U.dof_loc[::2,1]).T.reshape(-1)
        u[bdof] = u_err[bdof]
        p1[0] = p_exact(P1.dof_loc[0,0], P1.dof_loc[0,1]) # need to fix this pressure dof
        
        print("dimension = {}".format(fdof.sum()))

        # Homogeneize and then solve the system
        # sol_vec = group_fn(u, p1)
        sol_vec = group_fn(u, p1, p0)
        La = La - Aa @ sol_vec
        sol_vec_free = spsolve(Aa[fdof][:,fdof], La[fdof])
        sol_vec[fdof] = sol_vec_free
        # split_fn(sol_vec, u, p1)
        split_fn(sol_vec, u, p1, p0)

        # calculate the error
        u_err = u - u_err
        weak_error_table["u infty"][m] = np.linalg.norm(u_err, ord=np.inf)
        weak_error_table["u L2"][m] = np.sqrt(L2.assemble(dx, u=u_err._interpolate(dx)))

        p_err = Function(P1)
        p_err[:] = p_exact(P1.dof_loc[:,0], P1.dof_loc[:,1])
        p_err = p1 - p_err
        # p_err -= integral_P1.assemble(dx, p1=p_err._interpolate(dx)) / 1.0
        p0 -= integral_P1P0.assemble(dx, p1=p_err._interpolate(dx), p0=p0._interpolate(dx)) / 1.0
        weak_error_table["p L2"][m] = np.sqrt(L2_P1P0.assemble(dx, p1=p_err._interpolate(dx), p0=p0._interpolate(dx)))
        
        # =======================================================================
        # Imposing the normal velocity strongly
        p1[:] = 0.0
        p0[:] = 0.0
        Aa = bmat(((A, -B1, -B0), (B1.T, None, None), (B0.T, None, None)), format="csc")
        La = group_fn(-L + F_TAU, p1, p0)
        
        # identify the fixed dofs
        dof_lr = np.where((U.dof_loc[:,0] < 1e-12) | (U.dof_loc[:,0] > 1-1e-12))[0]
        dof_tb_n = np.where((U.dof_loc[1::2,1] < 1e-12) | (U.dof_loc[1::2,1] > 1-1e-12))[0] * 2 + 1
        bdof = np.unique(np.concatenate((dof_lr, dof_tb_n)))
        fdof = group_dof((U, P1, P0), (bdof, np.array((0,)), np.array((0,))))
        u_err[:] = u_exact(U.dof_loc[::2,0], U.dof_loc[::2,1]).T.reshape(-1)
        u[:] = 0.0
        u[bdof] = u_err[bdof]
        p1[0] = p_exact(P1.dof_loc[0,0], P1.dof_loc[0,1]) # need to fix this pressure dof

        # homogeneize the system
        sol_vec = group_fn(u, p1, p0)
        La = La - Aa @ sol_vec
        sol_vec_free = spsolve(Aa[fdof][:,fdof], La[fdof])
        sol_vec[fdof] = sol_vec_free
        split_fn(sol_vec, u, p1, p0)

        # calculate the error
        u_err[:] = u - u_err
        strong_error_table["u infty"][m] = np.linalg.norm(u_err, ord=np.inf)
        strong_error_table["u L2"][m] = np.sqrt(L2.assemble(dx, u=u_err._interpolate(dx)))

        p_err[:] = p_exact(P1.dof_loc[:,0], P1.dof_loc[:,1])
        p_err = p1 - p_err
        p0 -= integral_P1P0.assemble(dx, p1=p_err._interpolate(dx), p0=p0._interpolate(dx)) / 1.0
        strong_error_table["p L2"][m] = np.sqrt(L2_P1P0.assemble(dx, p1=p_err._interpolate(dx), p0=p0._interpolate(dx)))

    print(Fore.GREEN + "\nConvergence of Nitsche method: " + Style.RESET_ALL)
    printConvergenceTable(mesh_table, weak_error_table)
    print(Fore.GREEN + "\nConvergence of strong method: " + Style.RESET_ALL)
    printConvergenceTable(mesh_table, strong_error_table)
    