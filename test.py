from fem.mesh import Mesh
from fem.element import *
from fem.dof import *
from matplotlib import pyplot

if __name__ == "__main__":
    mesh = Mesh()
    mesh.load("mesh/two-phase.msh")

    space = Dof(mesh, TriP2)
    pass
