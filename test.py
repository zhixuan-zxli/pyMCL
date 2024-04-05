from fem.mesh import *
from fem.element import *
from fem.funcspace import *
from fem.function import *
from fem.quadrature import *
from matplotlib import pyplot

if __name__ == "__main__":
    mesh = Mesh()
    mesh.load("mesh/unit_square.msh")
    i_mesh = mesh.view(1, (2,))
    space = FunctionSpace(mesh, e)
    i_space = FunctionSpace(i_mesh, VectorElement(LineP1, 2))
    # mea = CellMeasure(mesh)
    # u = MeshMapping(space)
    # q = Quadrature.getTable(RefTri, 3)
    # u._interpolate_cell(mea, q)
    pass
    