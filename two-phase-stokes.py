import numpy as np
from dolfin import *
from msh2xdmf import import_mesh
# from matplotlib import pyplot

def testStokes():
    # load the mesh
    bulk_mesh, boundary_marker, subdomain_marker, assoc_table = \
        import_mesh(prefix='two-phase', subdomains=True, tdim=2, gdim=2, directory='mesh')
    interface_mesh = MeshView.create(boundary_marker, assoc_table["interface"])
    interface_mesh.init_cell_orientations(Expression(("x[0] - 0.5", "x[1]"), degree = 1))

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
            return on_boundary and near(x[0], 0.0)
        # map the right boundary (x) to the left (y)
        def map(self, x, y):
            y[0] = x[0] - 1.0
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
      - Constant(1.0/phys["Ca"]) * dot(nu, xi) * dS + Constant(phys["beta"]/phys["ls"]) * u[0] * v[0] * ds(bottom_id)
    l = dot(Constant((0.0, 0.0)), v) * dx

    # assemble the linear system
    sol = Function(FuncSpace)
    # u_sol = Function(FuncSpaces["U"])
    # p1_sol = Function(FuncSpaces["P1"])
    # p0_sol = Function(FuncSpaces["P0"])

    system = assemble_mixed_system(a == l, sol, essentialBCs)
    # A_blocks = system[0]
    # rhs_blocks = system[1]

    # A = PETScNestMatrix(A_blocks)
    # L = Vector()
    # sol_vec = Vector()
    # A.init_vectors(L, rhs_blocks)
    # A.init_vectors(sol_vec, [u_sol.vector(), p1_sol.vector(), p0_sol.vector()])

    # solve(A, sol_vec, L) # LU solver

    # get the sub solution
    # FuncSpaceDims = np.array([fs.dim() for fs in FuncSpaces.values()])
    # FuncSpaceDims = np.cumsum(FuncSpaceDims)
    # u_sol.vector().set_local(sol_vec.get_local()[:FuncSpaceDims[0]])
    # u_sol.vector().apply("")
    # p1_sol.vector().set_local(sol_vec.get_local()[FuncSpaceDims[0]:FuncSpaceDims[1]])
    # p1_sol.vector().apply("")
    # p0_sol.vector().set_local(sol_vec.get_local()[FuncSpaceDims[1]:])
    # p0_sol.vector().apply("")

    pass

testStokes()
