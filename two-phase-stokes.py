import numpy as np
import dolfin
from msh2xdmf import import_mesh
# from matplotlib import pyplot

# load the mesh
gdim = 2
# bulk_mesh, boundary_marker, subdomain_marker, physical_table = import_mesh(prefix='two-phase', subdomains=True, tdim=gdim, gdim=gdim, directory="mesh")
# interface_mesh = dolfin.MeshView.create(boundary_marker, physical_table["interface"])
# interface_mesh.init_cell_orientations(dolfin.Expression(("x[0] - 0.5", "x[1]"), degree = 1))
bulk_mesh = dolfin.UnitSquareMesh(16, 16)

# define the symbols
dx = dolfin.Measure('dx', domain=bulk_mesh)
# dx = dolfin.Measure('dx', domain=bulk_mesh, subdomain_data=subdomain_marker)
# dl = dolfin.Measure('ds', domain=bulk_mesh, subdomain_data=boundary_marker)
# dS = dolfin.Measure('dx', domain=interface_mesh)
# n = dolfin.CellNormal(interface_mesh)
# print('Measure of interface as MeshView = {}'.format(dolfin.assemble(dolfin.Constant(1.0) * dS)))
# proj_mea = dolfin.dot(dolfin.Constant((0.0, 1.0)), n) * dS
# print('Projected measure = {}'.format(dolfin.assemble(proj_mea)))

# Define the periodic boundary condition
class PeriodicBoundary(dolfin.SubDomain):
    # identify the left boundary
    def inside(self, x, on_boundary):
        return on_boundary and dolfin.near(x[0], 0.0)
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
FuncSpaces["U"] = dolfin.VectorFunctionSpace(bulk_mesh, 'CG', 2, dim=gdim, constrained_domain = periodic_bc)
FuncSpaces["P1"] = dolfin.FunctionSpace(bulk_mesh, 'CG', 1, constrained_domain = periodic_bc)
FuncSpaces["P0"] = dolfin.FunctionSpace(bulk_mesh, 'DG', 0, constrained_domain = periodic_bc)
# FuncSpaces["X"] = dolfin.VectorFunctionSpace(interface_mesh, 'CG', 1, dim=gdim)
# FuncSpaces["K"] = dolfin.FunctionSpace(interface_mesh, 'CG', 1)
FuncSpace = dolfin.MixedFunctionSpace(*FuncSpaces.values())

# define the flow boundary conditions
noSlipBC = dolfin.DirichletBC(FuncSpace.sub_space(0), dolfin.Constant((0.0, 0.0)), onMeshBoundary)
# noSlipBC = dolfin.DirichletBC(FuncSpace.sub_space(0), dolfin.Constant((0.0, 0.0)), boundary_marker, physical_table["top"])
# noPenBC = dolfin.DirichletBC(FuncSpace.sub_space(0).sub(1), dolfin.Constant(0.0), boundary_marker, physical_table["bottom"])
# define the attach boundary condition
# attachBC = dolfin.DirichletBC(FuncSpace.sub_space(3).sub(1), dolfin.Constant(0.0), onMeshBoundary)

essentialBCs = [noSlipBC] #, noPenBC] #, attachBC]

# define the physical parameters
phys = {"Re":1.0, "Ca":0.1, "ls":0.1, "beta":0.1}

f = dolfin.Expression(("-2*pi*pi*sin(2*pi*x[1])*(1-2*sin(pi*x[0]))*(1+2*sin(pi*x[0]))", \
                       "-2*pi*pi*sin(2*pi*x[0])*(2*sin(pi*x[1])+1)*(2*sin(pi*x[1])-1)"), degree=3)


# define the variational problem
# (u, p1, p0, x, kappa) = dolfin.TrialFunctions(FuncSpace)
# (v, q1, q0, y, chi) = dolfin.TestFunctions(FuncSpace)
(u, p1, p0) = dolfin.TrialFunctions(FuncSpace)
(v, q1, q0) = dolfin.TestFunctions(FuncSpace)
a = dolfin.Constant(1.0/phys["Re"]) * dolfin.inner(dolfin.grad(u), dolfin.grad(v)) * dx - dolfin.div(v) * (p1+p0) * dx + dolfin.div(u) * (q1+q0) * dx 
l = dolfin.dot(f, v) * dx
# l = dolfin.Constant(0.0) * (q1 + q0) * dx
#   + dolfin.Constant(phys["beta"]/phys["ls"]) * u[0] * v[0] * dl(physical_table["bottom"])
#   + dolfin.Constant(1.0/phys["Ca"]) * kappa * dolfin.dot(n, v) * dS
# l = dolfin.Constant(phys["gamma"] * phys["kappa"]) * dolfin.dot(n('+'), v('+')) * dS # \
#     # + dolfin.dot(dolfin.Constant((0.0, 0.0)), v) * dx # a hack to fix the interior facet normal
# b = dolfin.inner(dolfin.grad(u), dolfin.grad(v))*dx + (p1+p0)*(q1+q0)*dx # the preconditioning matrix

# assemble the linear system
sol = dolfin.Function(FuncSpace)
# dolfin.solve(a == l, sol, essentialBCs)
system = dolfin.assemble_mixed_system(a == l, sol, essentialBCs)
matrix_blocks = system[0]
rhs_blocks = system[1]
# A, L = dolfin.assemble_system(a, l, flow_bcs)
# B, _ = dolfin.assemble_system(b, l, flow_bcs)

A = dolfin.PETScNestMatrix(matrix_blocks)
L = dolfin.Vector()
A.init_vectors(L, rhs_blocks)
A.convert_to_aij()

# set up the solver
sol_vec = dolfin.Vector()
solver = dolfin.LUSolver()
solver.solve(A, sol_vec, L)
FuncSpaceDims = [fs.dim() for fs in FuncSpaces.values()]
u_sol = dolfin.Function(FuncSpaces["U"])
u_sol.vector().set_local(sol_vec.get_local()[:FuncSpaceDims[0]])
# (u, p1, p0) = sol.split()
# solver = dolfin.KrylovSolver("gmres", "amg")
# solver.set_operators(A, B)
# solver.parameters["report"] = True

# create the null vector
# sol = dolfin.Function(AugTHSpace)
# nv1 = dolfin.Vector(sol.vector())
# nv0 = dolfin.Vector(sol.vector())
# AugTHSpace.sub(1).dofmap().set(nv1, 1.0)
# AugTHSpace.sub(2).dofmap().set(nv0, 1.0)
# nv1 *= 1.0 / nv1.norm("l2")
# nv0 *= 1.0 / nv0.norm("l2")

# attach the null space to the PETSc matrix
# null_space = dolfin.VectorSpaceBasis([nv1, nv0])
# dolfin.as_backend_type(A).set_nullspace(null_space)

# null_space.orthogonalize(L)

# solve the linear system

# Convert the P1+P0 pressure to a DG1 function


# output the solution
outfile = dolfin.XDMFFile('data/two-phase-u.xdmf')
outfile.write(u_sol)
outfile.close()
