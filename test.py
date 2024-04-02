from fem.mesh import *
from fem.element import *
from fem.fe import *
from fem.function import *
from fem.quadrature import *
from matplotlib import pyplot

if __name__ == "__main__":
    mesh = Mesh()
    mesh.load("mesh/unit_square.msh")
    space = FiniteElement(mesh, TriP1)
    mea = CellMeasure(mesh)
    u = MeshMapping(space)
    q = Quadrature.getTable(RefTri, 3)
    u._interpolate_cell(mea, q)
    pass
    