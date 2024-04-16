import numpy as np
from math import cos
from fem import *
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot


class PhysicalParameters:
    eta_2: float = 0.1
    mu_1: float = 10.0
    mu_2: float = 10.0
    mu_cl: float = 1.0e-3
    cosY: float = cos(np.pi*2.0/3)
    B: float = 1.0

class SolverParemeters:
    dt: float = 1.0/1024
    Te: float = 1.0/8
    startStep: int = 0
    stride: int = 1
    numChekpoint: int = 0
    vis: bool = True

@BilinearForm
def c_lap(w: QuadData, q: QuadData, x: QuadData) -> np.ndarray:
    return w.grad[0] * q.grad[0] * x.dx

@BilinearForm
def c_H0(w: QuadData, q: QuadData, x: QuadData) -> np.ndarray:
    return w * q * x.dx

@BilinearForm
def c_cl(w: QuadData, q: QuadData, _, xq: QuadData) -> np.ndarray:
    return w * (q.grad[0] * xq.fn) * xq.ds

@LinearForm
def l_ver(phi: QuadData, x: QuadData) -> np.ndarray:
    return 0.5 * phi * x.ds


if __name__ == "__main__":

    phyp = PhysicalParameters()
    solp = SolverParemeters()

    # physical groups from GMSH
    # group_name = {"fluid_1": 1, "fluid_2": 2, "interface": 3, "dry": 4, "wet": 5, \
    #              "right": 6, "top": 7, "left": 8, "cl": 9, "clamp": 10}
    mesh = Mesh()
    mesh.load("mesh/two-phase.msh")
    # setMeshMapping(mesh, 2)
    # interface mesh
    # i_mesh = mesh.view(Measure(1, (3,)))
    # setMeshMapping(i_mesh)
    # sheet mesh
    s_mesh = mesh.view(1, sub_ids=(4,5))
    setMeshMapping(s_mesh)

    # set up the finite element spaces
    mixed_fs = [
        FunctionSpace(s_mesh, LineP2), # vertical displacement
        None            # moment
    ]
    mixed_fs[1] = mixed_fs[0]

    # set up the measures and the basis
    ds = Measure(s_mesh, 1, order=5)
    dp = Measure(s_mesh, 0, order=1, tags=(9,), interiorFacet=True) # at the contact line

    q_basis = FunctionBasis(mixed_fs[0], ds)
    q_cl_basis = FunctionBasis(mixed_fs[0], dp)

    # get the essential BC dof
    bdof = np.unique(mixed_fs[0].getFacetDof(tags=(10, )))
    fdof = group_dof(mixed_fs, (bdof, bdof))

    # assemble the system
    C_H0 = c_H0.assemble(q_basis, q_basis, ds)
    C_LAP = c_lap.assemble(q_basis, q_basis, ds)
    C_CL = c_cl.assemble(q_cl_basis, q_cl_basis, dp)
    L = l_ver.assemble(q_cl_basis, dp)

    Ca = bmat((
        (None, C_LAP - C_CL), # q
        (C_LAP - C_CL, 1.0/phyp.B*C_H0) # m
    ), format="csr")
    # Ca = bmat((
    #     (None, C_LAP), # q
    #     (C_LAP, 1.0/phyp.B*C_H0) # m
    # ), format="csr")
    
    # homogenize the system
    q = Function(mixed_fs[0])
    m = Function(mixed_fs[1])
    La = group_fn(L, m)
    # since the essential boundary conditions are homogeneous, 
    # no need to modify La
    sol = np.zeros_like(La)

    # solve the system
    sol_free = spsolve(Ca[fdof][:,fdof], La[fdof])
    sol[fdof] = sol_free
    split_fn(sol, q, m)

    # visualize the deformation
    pyplot.plot(mixed_fs[0].dof_loc[:,0], q, 'ro', label="deflection")
    pyplot.plot(mixed_fs[1].dof_loc[:,0], m, 'bx', label="moment")
    pyplot.legend()
    pyplot.axis("equal")
    pyplot.show()
    
