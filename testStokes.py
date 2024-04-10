import numpy as np
from fem.mesh import Mesh
from fem.mesh_util import splitRefine, setMeshMapping
from fem.element import Measure, TriDG0, TriP1, TriP2
from fem.function import Function, split_fn, group_fn
from fem.form import assembler, Form
from scipy import sparse
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot
from fem.post import printConvergenceTable

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
    return np.exp(np.pi*x) * np.cos(np.pi*y)

def dp_exact(x, y) -> np.ndarray:
    return np.array(
        (np.pi * np.exp(np.pi*x) * np.cos(np.pi*y), 
         -np.pi * np.exp(np.pi*x) * np.sin(np.pi*y))
    )

def f_exact(x, y) -> np.ndarray:
    diffu = np.array(
        (-2.0*np.pi**2 * np.sin(2.0*np.pi*y) * (1.0 + 2.0*np.sin(np.pi*x)) * (1.0 - 2.0*np.sin(np.pi*x)), 
         -2.0*np.pi**2 * np.sin(2.0*np.pi*x) * (2.0*np.sin(np.pi*y) + 1.0) * (2.0*np.sin(np.pi*y) - 1.0))
    )
    return diffu - dp_exact(x, y)

def a(v, u, coord) -> np.ndarray:
    # grad: (2, 2, Ne, Nq)
    z = np.zeros_like(u.grad)
    z[0,0,:,:] = 2.0 * v.grad[0,0,:,:] * u.grad[0,0,:,:] + v.grad[0,1,:,:] * u.grad[0,1,:,:]
    z[1,0,:,:] = v.grad[1,0,:,:] * u.grad[0,1,:,:]
    z[0,1,:,:] = v.grad[0,1,:,:] * u.grad[1,0,:,:]
    z[1,1,:,:] = v.grad[1,0,:,:] * u.grad[1,0,:,:] + 2.0 * v.grad[1,1,:,:] * u.grad[1,1,:,:]
    return z * coord.dx[np.newaxis]

def b(v, p, coord) -> np.ndarray:
    # v.grad: (2,2,Ne,Nq)
    z = np.zeros((2, 1, coord.shape[1], coord.shape[2]))
    z[0,0,:,:] = v.grad[0,0,:,:] * p[0,:,:]
    z[1,0,:,:] = v.grad[1,1,:,:] * p[0,:,:]
    return z * coord.dx[np.newaxis]

def l(v, coord) -> np.ndarray:
    return f_exact(coord[0], coord[1]) * v * coord.dx

def g(v, coord) -> np.ndarray:
    x, y = coord[0], coord[1]
    du = du_exact(x, y) # (2, 2, Ne, Nq)
    Du = du + du.transpose((1,0,2,3)) # 2 times sym grad
    p = p_exact(x, y) # (Ne, Nq)
    # coord.n : (2, Ne, Nq)
    Tn = np.sum(Du * coord.n[np.newaxis], axis=1) + p[np.newaxis] * coord.n # (2, Ne, Nq)
    # v : (2, 1, Nq)
    return Tn * v * coord.dx

def integral_P1P0(coord, p1, p0) -> np.ndarray:
    return (p1 + p0) * coord.dx

def L2_P1P0(coord, p1, p0) -> np.ndarray:
    return np.sum((p1+p0)**2, axis=0, keepdims=True) * coord.dx

def L2(coord, u) -> np.ndarray:
    return np.sum(u**2, axis=0, keepdims=True) * coord.dx


if __name__ == "__main__":

    num_hier = 3
    mesh_table = tuple(f"{i}" for i in range(num_hier))
    # error_head = ("u infty", "u L2", "p infty", "p L2")
    error_head = ("u infty", "u L2", "p L2")
    error_table = {k: [1.0] * num_hier for k in error_head}
    mesh = Mesh()

    for m in range(num_hier):
        print(f"Testing level {m}... ", end="")
        if m == 0:
            mesh.load("mesh/unit_square.msh")
        else:
            mesh = splitRefine(mesh)
        # Affine mesh
        setMeshMapping(mesh)
        # mesh boundary: left=6, right=7, bottom=8, top=9

        u_space, p1_space, p0_space = TriP2(mesh, 2), TriP1(mesh), TriDG0(mesh)

        # assemble the form
        asm_uu = assembler(u_space, u_space, Measure(2), order=3)
        asm_p = assembler(p1_space, None, Measure(2), order=3)

        A = asm_uu.bilinear(Form(a, "grad"))
        L = asm_uu.linear(Form(l, "f"))
        B1 = assembler(u_space, p1_space, Measure(2), order=3).bilinear(Form(b, "f", "grad"))
        B0 = assembler(u_space, p0_space, Measure(2), order=2).bilinear(Form(b, "f", "grad"))
        G = assembler(u_space, None, Measure(1, (2,4)), 3).linear(Form(g, "f"))

        # assemble the saddle point system
        p1 = Function(p1_space)
        p0 = Function(p0_space)
        Aa = sparse.bmat(((A, B1, B0), (B1.T, None, None), (B0.T, None, None)), format="csr")
        # La = group_fn(L, p1, p0)
        La = group_fn(L+G, p1, p0)

        # impose the Dirichlet condition
        bdof = np.unique(u_space.getCellDof(Measure(1, (3,5))))
        fdof = np.ones((La.shape[0], ), dtype=np.bool8)
        fdof[bdof*2] = False
        fdof[bdof*2+1] = False
        fdof[2*u_space.num_dof] = False # fix the dofs for pressure
        fdof[2*u_space.num_dof + p1_space.num_dof] = False
        u_ex = u_exact(u_space.dofloc[:,0], u_space.dofloc[:,1]).T
        u = Function(u_space)
        u[bdof] = u_ex[bdof]
        p1[0] = p_exact(p1_space.dofloc[0,0], p1_space.dofloc[0,1]) # need to set this pressure dof <<<<<<<
        
        print("Dimension of saddle point system = {}".format(fdof.sum()))

        # Homogeneize and then solve the system
        sol_vec = group_fn(u, p1, p0)
        La = La - Aa @ sol_vec
        z = spsolve(Aa[fdof][:,fdof], La[fdof])
        sol_vec[fdof] = z

        # extract the solutions
        split_fn(sol_vec, u, p1, p0)

        # calculate the error
        u_err = u - u_ex
        error_table["u infty"][m] = np.linalg.norm(u_err.ravel(), ord=np.inf)
        error_table["u L2"][m] = np.sqrt(asm_uu.functional(Form(L2, "f"), u = u_err))

        p_ex = p_exact(p1_space.dofloc[:, 0], p1_space.dofloc[:, 1]).reshape(-1, 1)
        p_err = p1 - p_ex
        p_err -= asm_p.functional(Form(integral_P1P0, "f"), p1 = p_err, p0 = p0)

        # error_table["p infty"][m] = np.linalg.norm(p_err, ord=np.inf)
        error_table["p L2"][m] = np.sqrt(asm_p.functional(Form(L2_P1P0, "f"), p1 = p_err, p0 = p0))

    printConvergenceTable(mesh_table, error_table)
    