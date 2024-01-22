import numpy as np
from dolfin import *
from msh2xdmf import import_mesh
# from matplotlib import pyplot

# load the mesh
gdim = 2
bulk_mesh = UnitSquareMesh(16, 16)

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
FuncSpaces["U"] = VectorFunctionSpace(bulk_mesh, 'CG', 2, dim=gdim, constrained_domain = periodic_bc)
FuncSpaces["P1"] = FunctionSpace(bulk_mesh, 'CG', 1, constrained_domain = periodic_bc)
FuncSpaces["P0"] = FunctionSpace(bulk_mesh, 'DG', 0, constrained_domain = periodic_bc)
FuncSpace = MixedFunctionSpace(*FuncSpaces.values())

# define the flow boundary conditions
noSlipBC = DirichletBC(FuncSpace.sub_space(0), Constant((0.0, 0.0)), onMeshBoundary)

essentialBCs = [noSlipBC] #, noPenBC] #, attachBC]

# define the physical parameters
phys = {"Re":1.0, "Ca":0.1, "ls":0.1, "beta":0.1}

f = Expression(("-2*pi*pi*sin(2*pi*x[1])*(1-2*sin(pi*x[0]))*(1+2*sin(pi*x[0]))", \
                       "-2*pi*pi*sin(2*pi*x[0])*(2*sin(pi*x[1])+1)*(2*sin(pi*x[1])-1)"), degree=3)


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

system = assemble_mixed_system(b == l, sol, essentialBCs)
B_blocks = system[0]
B_blocks[0] = A_blocks[0]
B = PETScNestMatrix(B_blocks)

# set up the preconditioner

# PETScOptions.set("ksp_view");
# PETScOptions.set("ksp_monitor_true_residual");
PETScOptions.set("pc_type", "fieldsplit");
PETScOptions.set("pc_fieldsplit_type", "additive");

PETScOptions.set("fieldsplit_0_ksp_type", "preonly");
PETScOptions.set("fieldsplit_0_pc_type", "gamg");
PETScOptions.set("fieldsplit_1_ksp_type", "preonly");
PETScOptions.set("fieldsplit_1_pc_type", "lu");
PETScOptions.set("fieldsplit_2_ksp_type", "preonly");
PETScOptions.set("fieldsplit_2_pc_type", "lu");

# solve the linear system
solver = PETScKrylovSolver("minres")
solver.set_from_options()
solver.set_operators(A, B)
prms = solver.parameters
prms["monitor_convergence"] = True
# prms["relative_tolerance"] = 1e-5

fields = [PETScNestMatrix.get_block_dofs(A, i) for i in range(3)]
PETScPreconditioner.set_fieldsplit(solver, fields, ["0", "1", "2"])

solver.solve(sol_vec, L)
sol_vec.apply("")

# get the sub solution
FuncSpaceDims = [fs.dim() for fs in FuncSpaces.values()]
u_sol.vector().set_local(sol_vec.get_local()[:FuncSpaceDims[0]])
u_sol.vector().apply("")

# Convert the P1+P0 pressure to a DG1 function

# output the solution
outfile = XDMFFile('data/two-phase-u.xdmf')
outfile.write(u_sol)
outfile.close()
