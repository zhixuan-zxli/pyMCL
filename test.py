from fem.mesh import *
from fem.element import *
from fem.measure import *
from fem.funcspace import *
from fem.function import *
from fem.mesh_util import *
from fem.form import *
from matplotlib import pyplot

@Form
def L2(x: QuadData, u: QuadData) -> np.ndarray:
    # x: (gdim, Ne, Nq)
    # x.dx: (1, Ne, Nq)
    # u: (1, Ne, Nq)
    # u.grad: (1, 2, Ne, Nq)
    return u**2 * x.dx
    # return x.dx

@Form
def L2s(x: tuple[QuadData], u: QuadData) -> np.ndarray:
    x, u = x[0], u[0]
    # u: (1, Ne, Nq)
    # x.ds: (1, Ne, Nq)
    return u**2 * x.ds

def u_exact(x, y) -> np.ndarray:
    return np.sin(np.pi*x) * np.cos(y)

if __name__ == "__main__":
    mesh = Mesh()
    mesh.load("mesh/unit_square.msh")
    setMeshMapping(mesh, 2)

    fs = FunctionSpace(mesh, TriP2)
    u = Function(fs)
    u[:] = u_exact(fs.dof_loc[:,0], fs.dof_loc[:,1])

    dx = Measure(mesh, 2, order=3)
    u_ = u._interpolate(dx)
    print("L2 integral = {}".format(L2.functional(dx, u=u_)))
    print("exact = {}".format(0.5 * (0.5 + 0.25*np.sin(2.))))

    ds = Measure(mesh, 1, order=3, tags=(4,))
    u_ = u._interpolate(ds)
    print("L2 surface integral / cos(1.)^2 = {}".format(L2s.functional(ds, u=u_) / np.cos(1.0)**2))

    