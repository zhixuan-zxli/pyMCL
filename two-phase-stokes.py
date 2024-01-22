import numpy as np
from dolfin import *
from msh2xdmf import import_mesh
from test_util import *
# from matplotlib import pyplot

def testStokes():
    # load the mesh
    bulk_mesh, boundary_marker, subdomain_marker, assoc_table = \
        import_mesh(prefix='two-phase', subdomains=True, tdim=2, gdim=2, directory='mesh')
    interface_mesh = MeshView.create(boundary_marker, assoc_table["interface"])
    interface_mesh.init_cell_orientations(Expression(("x[0]", "x[1]"), degree = 1))

    # define the symbols
    dx = Measure('dx', domain=bulk_mesh, subdomain_data=subdomain_marker)
    ds = Measure('ds', domain=bulk_mesh, subdomain_data=boundary_marker)
    dS = Measure('dx', domain=interface_mesh)
    dp = Measure('ds', domain=interface_mesh)
    n = CellNormal(interface_mesh)
    # just to check orientation
    proj_mea = dot(Constant((0.0, 1.0)), n) * dS
    print('Projected measure of interface = {}'.format(assemble(proj_mea)))

    # Define the periodic boundary condition
    class PeriodicBoundary(SubDomain):
        # identify the left boundary
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], -1.0)     # change this <<<<<<<<<<<<<
        # map the right boundary (x) to the left (y)
        def map(self, x, y):
            y[0] = x[0] - 2.0
            y[1] = x[1]
    periodic_bc = PeriodicBoundary()

    # define the function spaces
    FuncSpaces = dict()
    FuncSpaces["U"] = VectorFunctionSpace(bulk_mesh, 'CG', 2, constrained_domain = periodic_bc)
    FuncSpaces["P1"] = FunctionSpace(bulk_mesh, 'CG', 1, constrained_domain = periodic_bc)
    FuncSpaces["P0"] = FunctionSpace(bulk_mesh, 'DG', 0, constrained_domain = periodic_bc)
    FuncSpaces["H"] = FunctionSpace(interface_mesh, interface_mesh.ufl_coordinate_element())
    FuncSpace = MixedFunctionSpace(*FuncSpaces.values())

    # get the interface parametrization
    x = Function(FuncSpaces['H'])
    get_coordinates(x, interface_mesh.geometry())

    # define the flow boundary conditions
    bottom_id = assoc_table["bottom"]
    noSlipBC = DirichletBC(FuncSpace.sub_space(0), Constant((0.0, 0.0)), boundary_marker, assoc_table["top"])
    noPenBC = DirichletBC(FuncSpace.sub_space(0).sub(1), Constant(0.0), boundary_marker, assoc_table["bottom"])

    # Define the interface boundary condition -- it's Neumann

    essentialBCs = [noSlipBC, noPenBC] #, attachBC]

    # define the physical parameters
    phys = {"Re":1.0, "Ca":0.1, "ls":0.1, "beta":0.1}

    # define the variational problem
    (u, p1, p0, nu) = TrialFunctions(FuncSpace)
    (v, q1, q0, xi) = TestFunctions(FuncSpace)
    a = Constant(1.0/phys["Re"]) * inner(grad(u), grad(v)) * dx + div(v)*(p1+p0) * dx + div(u)*(q1+q0) * dx \
      - Constant(1.0/phys["Ca"]) * dot(nu, v) * dS + Constant(phys["beta"]/phys["ls"]) * u[0] * v[0] * ds(bottom_id) \
      + dot(nu, xi) * dS
    l = dot(Constant((0.0, 0.0)), v) * dx - inner(grad(x), grad(xi)) * dS + dot(Constant((0.0, -1.0)), xi) * dp

    # assemble the linear system
    sol = Function(FuncSpace)
    subSol = [Function(v) for v in FuncSpaces.values()]

    system = assemble_mixed_system(a == l, sol, essentialBCs)
    A_blocks = system[0]
    rhs_blocks = system[1]

    A = PETScNestMatrix(A_blocks)
    L = Vector()
    sol_vec = Vector()
    A.init_vectors(L, rhs_blocks)
    A.init_vectors(sol_vec, [s.vector() for s in subSol])

    A.convert_to_aij()
    solve(A, sol_vec, L) # use LU solver

    # get the sub solution
    FuncSpaceDims = np.array([0] + [v.dim() for v in FuncSpaces.values()])
    FuncSpaceDims = np.cumsum(FuncSpaceDims)
    for i in range(4):
        subSol[i].vector().set_local(sol_vec.get_local()[FuncSpaceDims[i]:FuncSpaceDims[i+1]])
        subSol[i].vector().apply("")

    # convert the pressure to DG1
    DG1_space = FunctionSpace(bulk_mesh, 'DG', 1)
    p1_proj = project(subSol[1], DG1_space)
    p0_proj = project(subSol[2], DG1_space)
    p1_proj.vector().add_local(p0_proj.vector().get_local())
    p1_proj.vector().apply("")
    zeroMean(p1_proj, dx)

    # output the solution
    subSol[0].rename("u", "velocity")
    with XDMFFile("data/two-phase-u.xdmf") as outfile:
        outfile.write(subSol[0])
    p1_proj.rename("p", "pressure")
    with XDMFFile("data/two-phase-p.xdmf") as outfile:
        outfile.write(p1_proj)
    subSol[3].rename("nu", "mean_curavature_vector")
    with XDMFFile("data/two-phase-nu.xdmf") as outfile:
        outfile.write(subSol[3])

testStokes()
