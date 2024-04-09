from fem.mesh import *
from fem.element import *
from fem.measure import *
from fem.funcspace import *
from fem.function import *
from fem.mesh_util import *
from fem.form import *
from matplotlib import pyplot

@Form
def L2(x: QuadData, u: QuadData, v: QuadData) -> np.ndarray:
    # x: (gdim, Ne, Nq)
    # x.dx: (1, Ne, Nq)
    # u: (1, Ne, Nq)
    # u.grad: (1, 2, Ne, Nq)
    return u * v * x.ds

def u_exact(x, y) -> np.ndarray:
    return np.sin(np.pi*x) * np.cos(y)

if __name__ == "__main__":
    mesh = Mesh()
    mesh.load("mesh/unit_square.msh")
    setMeshMapping(mesh)

    b_mesh = mesh.view(1, (4,))
    setMeshMapping(b_mesh)

    fs2 = FunctionSpace(mesh, TriP2)
    u = Function(fs2)
    u[:] = u_exact(fs2.dof_loc[:,0], fs2.dof_loc[:,1])

    fs1 = FunctionSpace(b_mesh, LineP2)
    v = Function(fs1)
    v[:] = u_exact(fs1.dof_loc[:,0], fs1.dof_loc[:,1])

    ds = Measure(mesh, 1, order=3, tags=(4,))
    dx = Measure(b_mesh, 1, order=3)

    r = L2.assemble(ds, u=u._interpolate(ds), v=v._interpolate(dx))
    print("L2 surface integral / cos(1.)^2 = {}".format(r / np.cos(1.0)**2))

    