import numpy as np
import fem
from element import Lagrange
from matplotlib import pyplot

if __name__ == "__main__":
    mesh = fem.Mesh()
    mesh.load("mesh/unit_square.msh")
    u_space = fem.FESpace(mesh, Lagrange("tri", 2))
    pass
