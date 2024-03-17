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

def rhs(psi, coord) -> np.ndarray:
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
        # Affine mesh
        setMeshMapping(mesh)
        mesh.add_constraint(lambda x: np.abs(x[:,0]) < 1e-14, 
                            lambda x: np.abs(x[:,0]-1.0) < 1e-14, 
                            lambda x: x + np.array(((1.0, 0.0))), 
                            tol=1e-11)
        # mesh.add_constraint(lambda x: np.abs(x[:,1]) < 1e-14, 
        #                     lambda x: np.abs(x[:,1]-1.0) < 1e-14, 
        #                     lambda x: x + np.array(((0.0, 1.0))), 
        #                     tol=1e-11)
        
        # fe = TriP1(mesh, 1, periodic=False)
        fe = TriP1(mesh, 1, periodic=True)

        asm_2 = assembler(fe, fe, Measure(2), order=3)
        f = asm_2.linear(Form(rhs, "f"))
        A = asm_2.bilinear(Form(a, "grad"))

        # get freeze dof
        # dirichlet_dof = [0]
        dirichlet_dof = np.unique(fe.getCellDof(Measure(1, (2, 4))))
        # dirichlet_dof = np.unique(fe.getCellDof(Measure(1, (2, 3, 4, 5))))
        slave_dof = np.nonzero(fe.dof_remap != np.arange(fe.num_dof))[0]
        freeze_dof = np.unique(np.concatenate((dirichlet_dof, slave_dof)))
        free_dof = np.ones((fe.num_dof, ), dtype=np.bool8)
        free_dof[freeze_dof] = False
        # free_dof[dirichlet_dof] = False

        u_e = Function(fe)
        u_e[:,0] = exact(fe.dofloc[:,0], fe.dofloc[:,1])
        u = Function(fe)
        u[dirichlet_dof] = u_e[dirichlet_dof]

        # homogeneize the system
        f = f - A @ u
        sol_vec = spsolve(A[free_dof][:,free_dof], f[free_dof])
        u[free_dof,0] = sol_vec
        u.update()

        u_e -= u
        
        # ax_err = fig.add_subplot(1, 2, 2, projection="3d")
        # ax_err.plot_trisurf(mesh.point[:,0], mesh.point[:,1], u_err.ravel(), triangles=mesh.cell[2][:,:-1], cmap=pyplot.cm.Spectral)
        # pyplot.show()
        error_table["infty"][m] = np.linalg.norm(u_e, ord=np.inf)
        error_table["L2"][m] = sqrt(asm_2.functional(Form(L2, "f"), u=u_e))

    printConvergenceTable(mesh_table, error_table)
    