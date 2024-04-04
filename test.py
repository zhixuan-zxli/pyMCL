from fem.mesh import *
from fem.element import *
from fem.funcspace import *
from fem.function import *
from fem.quadrature import *
from matplotlib import pyplot

if __name__ == "__main__":
    mesh = Mesh()
    mesh.load("mesh/two-phase.msh")
    i_mesh = mesh.view(1, (3,))
    e = VectorElement(TriP2, 2)
    print(e.dof_name)
    space = FunctionSpace(mesh, TriP2)
    # mea = CellMeasure(mesh)
    # u = MeshMapping(space)
    # q = Quadrature.getTable(RefTri, 3)
    # u._interpolate_cell(mea, q)
    pass
    