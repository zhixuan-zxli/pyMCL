from fem.mesh import *
from fem.element import *
from fem.measure import *
from fem.funcspace import *
from fem.function import *
from fem.mesh_util import *
from fem.form import *
from matplotlib import pyplot

@Form
def dx(x: np.ndarray) -> np.ndarray:
    # x: (gdim, Ne, Nq)
    # x.dx: (1, Ne, Nq)
    return x.dx

if __name__ == "__main__":
    mesh = Mesh()
    mesh.load("mesh/two-phase.msh")
    setMeshMapping(mesh)

    # i_mesh = mesh.view(1, (2,))
    # setMeshMapping(i_mesh)

    # space = FunctionSpace(mesh, TriP1)
    dx_1 = Measure(mesh, 2, order=3, tags=(1,))
    dx_2 = Measure(mesh, 2, order=3, tags=(2,))
    
    print("Domain area = {}, {}".format(dx.functional(dx_1), dx.functional(dx_2)))

    