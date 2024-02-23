import numpy as np
from Mesh import Mesh

if __name__ == "__main__":
    mesh = Mesh()
    mesh.load("mesh/two-phase.msh")
    interface_mesh = mesh.view(1, [3])
    print("Finished.")
