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

def exact(x, y) -> np.ndarray:
    return np.sin(np.pi*x) * np.cos(y)

def exact_grad(x, y) -> np.ndarray:
    return np.array((np.pi * np.cos(np.pi*x) * np.cos(y), -np.sin(np.pi*x) * np.sin(y)))

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

def bc(psi, coord) -> np.ndarray:
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

    num_hier = 3
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
        # P2 Isoparametric mapping
        # coord_fe = TriP2(mesh, num_copy=mesh.gdim)
        # coord_map = Function(coord_fe)
        # coord_map[:] = coord_fe.dofloc
        # setMeshMapping(mesh, coord_map)

        fe = TriP2(mesh)

        asm_2 = assembler(fe, fe, Measure(2, None), order=4)
        f = asm_2.linear(Form(rhs, "f"))
        A = asm_2.bilinear(Form(a, "grad"))
        v = asm_2.linear(Form(test, "f"))

        asm_1 = assembler(fe, None, Measure(1, (6,7,8,9)), order=3, geom_hint=("f", "grad", "dx", "n"))
        g = asm_1.linear(Form(bc, "f"))

        # build the augmented system
        Aa = sparse.bmat(((A, v),(v.T, 0.0)), format="csr")
        fg = np.vstack((f+g, 0.0))

        u_sol = spsolve(Aa, fg).reshape(-1, 1)
        # print("lambda = {}".format(u_sol[-1,0]))
        u = Function(fe)
        u[:] = u_sol[:-1]

        u_err = Function(fe)
        # u_err[:] = exact(mesh.point[:, 0], mesh.point[:, 1]).reshape(-1, 1) # For P1
        u_err[:] = exact(fe.dofloc[:, 0], fe.dofloc[:, 1]).reshape(-1, 1) # For P2
        u_err -= u
        u_err -= asm_2.functional(Form(integral, "f"), u=u_err) / 1.0
        
        # ax_err = fig.add_subplot(1, 2, 2, projection="3d")
        # ax_err.plot_trisurf(mesh.point[:,0], mesh.point[:,1], u_err.ravel(), triangles=mesh.cell[2][:,:-1], cmap=pyplot.cm.Spectral)
        # pyplot.show()
        error_table["infty"][m] = np.linalg.norm(u_err, ord=np.inf)
        error_table["L2"][m] = np.sqrt(asm_2.functional(Form(L2, "f"), u=u_err))

    printConvergenceTable(mesh_table, error_table)
    