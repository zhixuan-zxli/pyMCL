import numpy as np
from mesh import Mesh
from fe import Measure, TriP1, TriP2
from function import Function
from assemble import assembler, Form, setMeshMapping
from scipy.sparse.linalg import spsolve
# from matplotlib import pyplot

def exact(x, y) -> np.ndarray:
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    return np.sin(np.pi*x) * np.cos(y)

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

if __name__ == "__main__":
    mesh = Mesh()
    mesh.load("mesh/unit_square.msh")
    setMeshMapping(mesh)
    u_space = TriP1(mesh)
    asm = assembler(u_space, u_space, Measure(2, None), 3)
    f = asm.linear(Form(rhs, "f"))
    A = asm.bilinear(Form(a, "grad"))
    # collect boundary dofs
    bdof = u_space.getCellDof(Measure(1, (2,)))
    bdof = np.unique(bdof)
    fdof = np.ones((u_space.num_dof, ), dtype=np.bool8)
    fdof[bdof] = False
    # interpolate
    u_exact = exact(mesh.point[:,0], mesh.point[:,1]) # (Np,)
    u = Function(u_space)
    u[0, bdof] = u_exact[0, bdof]
    # homogeneize
    f = f - A @ u
    # solve
    u_sol = spsolve(A[fdof,fdof], u[fdof], use_umfpack=True)
    # find error
    u[fdof] = u_sol
    u_err = u - u_exact
    print("error = {0:.3e}", np.linalg.norm(u_err, ord=np.inf))