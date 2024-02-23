import numpy as np
# from Mesh import Mesh
import fem

if __name__ == "__main__":
    mesh = fem.Mesh()
    mesh.load("mesh/two-phase.msh")
    interface_mesh = mesh.view(1, [3])
    u_space = fem.FESpace(mesh, fem.Element("Lagrange", 2))
