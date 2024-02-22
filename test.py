import numpy as np
from Mesh import Mesh

mesh = Mesh("mesh/two-phase.msh")
mesh.buildP2Mesh()
print("Finished.")
