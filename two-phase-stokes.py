import numpy as np
from dolfin import *
from msh2xdmf import import_mesh
# from matplotlib import pyplot

mesh_table = [8, 16, 32, 64]
error_table = {"u, L2":[], 
               "u, H10":[], 
               "u, Linf":[], 
               "p, L2":[], 
               "p, Linf":[]}

def printConvergenceTable(mesh_table, error_table):
    m = len(mesh_table)
    # print the header
    header_str = "\n{0: <20}".format("")
    for i in range(m-1):
        header_str += "{0: <10}{1: <8}".format(mesh_table[i], "rate")
    header_str += "{0: <10}".format(mesh_table[-1])
    print(header_str)
    # print each norm
    for (norm_type, error_list) in error_table.items():
        error_str = "{0: <20}".format(norm_type)
        for i in range(m-1):
            error_str += "{0:<10.2e}{1:<8.2f}".format(error_list[i], np.log2(error_list[i]/error_list[i+1]))
        error_str += "{0:<10.2e}".format(error_list[-1])
        print(error_str)

def zeroMean(p, dx):
    p_avr = assemble(p*dx) / assemble(Constant(1.0) * dx)
    p_vec = p.vector().get_local() - p_avr
    p.vector().set_local(p_vec)
    p.vector().apply("")
    return p

def testStokes(mesh_div):
    # load the mesh
    bulk_mesh = UnitSquareMesh(mesh_div, mesh_div)

    # define the symbols
    dx = Measure('dx', domain=bulk_mesh)

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

    # Define the endpoint of the interface
    def onMeshBoundary(x, on_boundary):
        return on_boundary

    # define the function spaces
    FuncSpaces = dict()
    FuncSpaces["U"] = VectorFunctionSpace(bulk_mesh, 'CG', 2, constrained_domain = periodic_bc)
    FuncSpaces["P1"] = FunctionSpace(bulk_mesh, 'CG', 1, constrained_domain = periodic_bc)
    FuncSpaces["P0"] = FunctionSpace(bulk_mesh, 'DG', 0, constrained_domain = periodic_bc)
    FuncSpace = MixedFunctionSpace(*FuncSpaces.values())

    # define the flow boundary conditions
    noSlipBC = DirichletBC(FuncSpace.sub_space(0), Constant((0.0, 0.0)), onMeshBoundary)

    essentialBCs = [noSlipBC] #, noPenBC] #, attachBC]

    # define the physical parameters
    phys = {"Re":1.0, "Ca":0.1, "ls":0.1, "beta":0.1}

    f = Expression(("-2*pi*pi*sin(2*pi*x[1])*(1-2*sin(pi*x[0]))*(1+2*sin(pi*x[0])) + 2*pi*sin(2*pi*x[0])*exp(pi*x[1])", \
                        "-2*pi*pi*sin(2*pi*x[0])*(2*sin(pi*x[1])+1)*(2*sin(pi*x[1])-1) - pi * cos(2*pi*x[0]) * exp(pi*x[1])"), degree=3)

    u_exact = Expression(("sin(pi*x[0])*sin(pi*x[0])*sin(2*pi*x[1])", "-sin(2*pi*x[0])*sin(pi*x[1])*sin(pi*x[1])"), degree=3)
    p_exact = Expression("cos(2*pi*x[0]) * exp(pi*x[1])", degree=3)

    # define the variational problem
    (u, p1, p0) = TrialFunctions(FuncSpace)
    (v, q1, q0) = TestFunctions(FuncSpace)
    a = Constant(1.0/phys["Re"]) * inner(grad(u), grad(v)) * dx + div(v) * (p1+p0) * dx + div(u) * (q1+q0) * dx 
    l = dot(f, v) * dx
    b = Constant(0.0) * inner(u, v) * dx + (p1+p0)*(q1+q0)*dx # the preconditioning matrix

    # assemble the linear system
    sol = Function(FuncSpace)
    u_sol = Function(FuncSpaces["U"])
    p1_sol = Function(FuncSpaces["P1"])
    p0_sol = Function(FuncSpaces["P0"])

    system = assemble_mixed_system(a == l, sol, essentialBCs)
    A_blocks = system[0]
    rhs_blocks = system[1]

    A = PETScNestMatrix(A_blocks)
    L = Vector()
    sol_vec = Vector()
    A.init_vectors(L, rhs_blocks)
    A.init_vectors(sol_vec, [u_sol.vector(), p1_sol.vector(), p0_sol.vector()])

    # A.convert_to_aij()

    system = assemble_mixed_system(b == l, sol, essentialBCs)
    B_blocks = system[0]
    B_blocks[0] = A_blocks[0]
    B = PETScNestMatrix(B_blocks)

    # set up the preconditioner

    # PETScOptions.set("ksp_view");
    # PETScOptions.set("ksp_monitor_true_residual");
    PETScOptions.set("pc_type", "fieldsplit");
    PETScOptions.set("pc_fieldsplit_type", "multiplicative");

    PETScOptions.set("fieldsplit_0_ksp_type", "preonly");
    PETScOptions.set("fieldsplit_0_pc_type", "hypre");
    PETScOptions.set("fieldsplit_1_ksp_type", "preonly");
    PETScOptions.set("fieldsplit_1_pc_type", "lu");
    PETScOptions.set("fieldsplit_2_ksp_type", "preonly");
    PETScOptions.set("fieldsplit_2_pc_type", "lu");

    # solve the linear system
    solver = PETScKrylovSolver("bicgstab")
    solver.set_from_options()
    solver.set_operators(A, B)
    prms = solver.parameters
    prms["monitor_convergence"] = True
    # prms["relative_tolerance"] = 1e-6
    # prms["report"] = True

    fields = [PETScNestMatrix.get_block_dofs(A, i) for i in range(3)]
    PETScPreconditioner.set_fieldsplit(solver, fields, ["0", "1", "2"])

    # solve(A, sol_vec, L) # LU solver
    solver.solve(sol_vec, L)
    sol_vec.apply("")

    # get the sub solution
    FuncSpaceDims = np.array([fs.dim() for fs in FuncSpaces.values()])
    FuncSpaceDims = np.cumsum(FuncSpaceDims)
    u_sol.vector().set_local(sol_vec.get_local()[:FuncSpaceDims[0]])
    u_sol.vector().apply("")
    p1_sol.vector().set_local(sol_vec.get_local()[FuncSpaceDims[0]:FuncSpaceDims[1]])
    p1_sol.vector().apply("")
    p0_sol.vector().set_local(sol_vec.get_local()[FuncSpaceDims[1]:])
    p0_sol.vector().apply("")

    # calculate the errors
    error_table["u, L2"].append(errornorm(u_exact, u_sol, "L2"))
    error_table["u, H10"].append(errornorm(u_exact, u_sol, "H10"))
    u_err = project(u_exact, FuncSpaces["U"])
    u_err.vector().axpy(-1.0, u_sol.vector())
    u_err.vector().apply("")
    error_table["u, Linf"].append(np.linalg.norm(u_err.vector().get_local(), np.inf))

    # Convert the P1+P0 pressure to a DG1 function
    DG1_space = FunctionSpace(bulk_mesh, 'DG', 1)
    p1_proj = project(p1_sol, DG1_space)
    p0_proj = project(p0_sol, DG1_space)
    p1_proj.vector().axpy(1.0, p0_proj.vector())
    p1_proj.vector().apply("")
    zeroMean(p1_proj, dx)

    # with XDMFFile("data/two-phase-p.xdmf") as outfile:
    #     outfile.write(p1_proj)

    # calculate the errors for the pressure
    p_err = project(p_exact, DG1_space)
    p_err.vector().axpy(-1.0, p1_proj.vector())
    p_err.vector().apply("")
    zeroMean(p_err, dx)
    error_table["p, L2"].append(np.sqrt(assemble(p_err**2*dx)))
    error_table["p, Linf"].append(np.linalg.norm(p_err.vector().get_local(), np.inf))


for i in mesh_table:
    print('Testing level {}...'.format(i))
    testStokes(i)
printConvergenceTable(mesh_table, error_table)
