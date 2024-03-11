import numpy as np
from mesh import Mesh
from mesh_util import splitRefine, setMeshMapping
from fe import Measure, TriDG0, TriP1, TriP2
from function import Function, split_fn, group_fn
from assemble import assembler, Form
from scipy import sparse
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot
from util import printConvergenceTable

def u_exact(x, y) -> np.ndarray:
    return np.array(
        (np.sin(np.pi*x)**2 * np.sin(2*np.pi*y), 
         -np.sin(2*np.pi*x) * np.sin(np.pi*y)**2)
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

def a(u, v, coord) -> np.ndarray:
    # grad: (2, 2, Ne, Nq)
    z = np.zeros_like(u.grad)
    z[0,0,:,:] = 2.0 * v.grad[0,0,:,:] * u.grad[0,0,:,:] + v.grad[0,1,:,:] * u.grad[0,1,:,:]
    z[1,0,:,:] = v.grad[1,0,:,:] * u.grad[0,1,:,:]
    z[0,1,:,:] = v.grad[0,1,:,:] * u.grad[1,0,:,:]
    z[1,1,:,:] = v.grad[1,0,:,:] * u.grad[1,0,:,:] + 2.0 * v.grad[1,1,:,:] * u.grad[1,1,:,:]
    return z * coord.dx[np.newaxis]

def b(p, v, coord) -> np.ndarray:
    # v.grad: (2,2,Ne,Nq)
    z = np.zeros((2, 1, coord.shape[1], coord.shape[2]))
    z[0,0,:,:] = v.grad[0,0,:,:] * p[0,:,:]
    z[1,0,:,:] = v.grad[1,1,:,:] * p[0,:,:]
    return z * coord.dx[np.newaxis]

def l(v, coord) -> np.ndarray:
    return f_exact(coord[0], coord[1]) * v * coord.dx

def integral(coord, u) -> np.ndarray:
    return u * coord.dx

def L2(coord, u) -> np.ndarray:
    return np.sum(u**2, axis=0, keepdims=True) * coord.dx


if __name__ == "__main__":

    num_hier = 3
    mesh_table = tuple(f"{i}" for i in range(num_hier))
    error_head = ("u infty", "u L2", "p infty", "p L2")
    error_table = {k: [1.0] * num_hier for k in error_head}
    mesh = Mesh()

    for m in range(num_hier):
        print(f"Testing level {m}...")
        if m == 0:
            mesh.load("mesh/unit_square.msh")
        else:
            mesh = splitRefine(mesh)
        # Affine mesh
        setMeshMapping(mesh)

        u_space, p_space = TriP2(mesh, 2), TriP1(mesh)

        # assemble the form
        asm_uu = assembler(u_space, u_space, Measure(2), order=3)
        asm_up = assembler(u_space, p_space, Measure(2), order=3)
        asm_p = assembler(p_space, None, Measure(2), order=3)
        A = asm_uu.bilinear(Form(a, "grad"))
        L = asm_uu.linear(Form(l, "f"))
        B = asm_up.bilinear(Form(b, "f", "grad"))

        # assemble the saddle point system
        p = Function(p_space)
        pp = sparse.csr_array((p.shape[0], p.shape[0]))
        Aa = sparse.bmat(((A, B), (B.T, pp)), format="csr")
        La = group_fn(L, p)

        print("Dimension of saddle point system = {}".format(Aa.shape))

        # impose the Dirichlet condition
        bdof = np.unique(u_space.getCellDof(Measure(1, (2, ))))
        fdof = np.ones((La.shape[0], ), dtype=np.bool8)
        fdof[bdof*2] = False
        fdof[bdof*2+1] = False
        fdof[2*u_space.num_dof] = False # fix the first dof for pressure

        u_ex = u_exact(u_space.dofloc[:,0], u_space.dofloc[:,1]).T
        u = Function(u_space)
        u[bdof, :] = u_ex[bdof, :]
        sol_vec = group_fn(u, p)
        Lah = La - Aa @ sol_vec
        z = spsolve(Aa[fdof][:,fdof], Lah[fdof])
        sol_vec[fdof, 0] = z

        # extract the solutions
        split_fn(sol_vec, u, p)

        # calculate the error
        u_err = u - u_ex
        p_ex = p_exact(p_space.dofloc[:, 0], p_space.dofloc[:, 1]).reshape(-1, 1)
        p_err = p - p_ex
        p_err -= asm_p.functional(Form(integral, "f"), u = p_err)

        error_table["u infty"][m] = np.linalg.norm(u_err.ravel(), ord=np.inf)
        error_table["u L2"][m] = np.sqrt(asm_uu.functional(Form(L2, "f"), u = u_err))
        error_table["p infty"][m] = np.linalg.norm(p_err, ord=np.inf)
        error_table["p L2"][m] = np.sqrt(asm_p.functional(Form(L2, "f"), u = p_err))

    printConvergenceTable(mesh_table, error_table)
    