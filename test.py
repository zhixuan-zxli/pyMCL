from fem.mesh import *
from fem.element import *
from fem.funcspace import *
from fem.function import *
from fem.measure import *
from fem.mesh_util import *
from fem.form import *
from matplotlib import pyplot

if __name__ == "__main__":
    mesh = Mesh()
    mesh.load("mesh/unit_square.msh")
    setMeshMapping(mesh)

    i_mesh = mesh.view(1, (2,))
    setMeshMapping(i_mesh)

    space = FunctionSpace(mesh, TriP1)
    mea = Measure(mesh, 2)

    asbr = assembler(space, None, mea, 3)
    pass
    