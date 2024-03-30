import numpy as np
from math import cos
from fem.mesh import Mesh, Measure
from fem.mesh_util import setMeshMapping
from fem.element import TriDG0, TriP1, TriP2, LineP1, LineP2, group_dof
from fem.function import Function, split_fn, group_fn
from fem.assemble import assembler, Form
from scipy import sparse
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot


class PhysicalParameters:
    eta_2: float = 0.1
    mu_1: float = 10.0
    mu_2: float = 10.0
    mu_cl: float = 1.0e-3
    cosY: float = cos(np.pi*2.0/3)

class SolverParemeters:
    dt: float = 1.0/1024
    Te: float = 1.0/8
    startStep: int = 0
    stride: int = 1
    numChekpoint: int = 0
    vis: bool = True


if __name__ == "__main__":

    phyp = PhysicalParameters()
    solp = SolverParemeters()

    # physical groups from GMSH
    # group_name = {"fluid_1": 1, "fluid_2": 2, "interface": 3, "dry": 4, "wet": 5, \
    #              "right": 6, "top": 7, "left": 8, "cl": 9, "clamp": 10}
    mesh = Mesh()
    mesh.load("mesh/two-phase.msh")
    mesh.add_constraint(lambda x: np.abs(x[:,0]+1.0) < 1e-14, 
                        lambda x: np.abs(x[:,0]-1.0) < 1e-14, 
                        lambda x: x + np.array((2.0, 0.0)), 
                        tol=1e-11)
    setMeshMapping(mesh, 2)
    # interface mesh
    i_mesh = mesh.view(Measure(1, (3,)))
    setMeshMapping(i_mesh)
    # sheet mesh
    s_mesh = mesh.view(Measure(1, (4,5)))
    setMeshMapping(s_mesh, 2)

    # set up the finite element spaces
    mixed_fe = [
        LineP2(s_mesh), # vertical displacement
        None            # moment
    ]
    mixed_fe[1] = mixed_fe[0]

    # get the wall dof and the cl dof
    #wall_dof = 