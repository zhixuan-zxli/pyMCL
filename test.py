import numpy as np
# from Mesh import Mesh
import Fem

if __name__ == "__main__":
    mesh = Fem.Mesh()
    mesh.load("mesh/two-phase.msh")
    interface_mesh = mesh.view(1, [3])
    u_space = Fem.FunctionSpace(mesh, Fem.Element.Lagrange2(), 2)
    pass
