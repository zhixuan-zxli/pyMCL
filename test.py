import numpy as np
# from Mesh import Mesh
import fem
from matplotlib import pyplot

if __name__ == "__main__":
    mesh = fem.Mesh()
    mesh.load("mesh/unit_square.msh")
    # pyplot.figure()
    # mesh.draw()
    # mesh_2 = mesh.refine()
    # pyplot.figure()
    # mesh_2.draw()
    # pyplot.show()
    # interface_mesh = mesh.view(1, [3])
    u_space = fem.FESpace(mesh, fem.Element("Lagrange", 2))
    pass
