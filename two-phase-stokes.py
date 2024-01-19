import numpy as np
import dolfin
from msh2xdmf import import_mesh
from matplotlib import pyplot

# load the mesh
mesh, boundary_marker, subdomain_marker, physical_table = import_mesh(prefix='two-phase', subdomains=True, tdim=2, gdim=2, directory="mesh")
# mesh = dolfin.UnitSquareMesh(16, 16)

# define the symbols
dx = dolfin.Measure('dx', domain=mesh, subdomain_data=subdomain_marker)
# ds = dolfin.Measure('ds', domain=mesh, subdomain_data=boundary_marker)
dS = dolfin.Measure('dS', domain=mesh, subdomain_id=physical_table["interface"], subdomain_data=boundary_marker)
n = dolfin.FacetNormal(mesh)
print('Interface measure = {}'.format(dolfin.assemble(dolfin.Constant(1.0) * dS)))
proj_mea = dolfin.dot(dolfin.Constant((0.0, 1.0)), n('+')) * dS
print('Projected measure = {}'.format(dolfin.assemble(proj_mea)))

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

# define the function spaces
cell = mesh.ufl_cell()
U = dolfin.VectorElement('CG', cell, degree=2)
P = dolfin.FiniteElement('CG', cell, degree=1)
TaylorHoodSpace = dolfin.FunctionSpace(mesh, dolfin.MixedElement([U, P]), constrained_domain = periodic_bc)

# define the flow boundary conditions
noSlipBC_T = dolfin.DirichletBC(TaylorHoodSpace.sub(0), dolfin.Constant((0.0, 0.0)), \
                              boundary_marker, physical_table["top"])
noPen_L = dolfin.DirichletBC(TaylorHoodSpace.sub(0).sub(1), dolfin.Constant(0.0), \
                                  boundary_marker, physical_table["dryleft"])
noPen_W = dolfin.DirichletBC(TaylorHoodSpace.sub(0).sub(1), dolfin.Constant(0.0), \
                                  boundary_marker, physical_table["wet"])
noPen_R = dolfin.DirichletBC(TaylorHoodSpace.sub(0).sub(1), dolfin.Constant(0.0), \
                                  boundary_marker, physical_table["dryright"])
flow_bcs = [noSlipBC_T, noPen_L, noPen_W, noPen_R]

# define the physical parameters
phys = {"gamma":1.0, "kappa":-0.2}

# define the variational problem
u, p = dolfin.TrialFunctions(TaylorHoodSpace)
v, q = dolfin.TestFunctions(TaylorHoodSpace)
a = dolfin.inner(dolfin.grad(u), dolfin.grad(v)) * dx + dolfin.div(v) * p * dx + dolfin.div(u) * q * dx
l = dolfin.Constant(phys["gamma"] * phys["kappa"]) * dolfin.dot(n('+'), v('+')) * dS # \
    # + dolfin.dot(dolfin.Constant((0.0, 0.0)), v) * dx # a hack to fix the interior facet normal
b = dolfin.inner(dolfin.grad(u), dolfin.grad(v))*dx + p*q*dx

# assemble the linear system
A, L = dolfin.assemble_system(a, l, flow_bcs)
B, _ = dolfin.assemble_system(b, l, flow_bcs)

# set up the solver
solver = dolfin.KrylovSolver("gmres", "amg")
solver.set_operators(A, B)
solver.parameters["report"] = True

# create the null vector
sol = dolfin.Function(TaylorHoodSpace)
null_vec = dolfin.Vector(sol.vector())
TaylorHoodSpace.sub(1).dofmap().set(null_vec, 1.0)
null_vec *= 1.0 / null_vec.norm("l2")

# attach the null space to the PETSc matrix
null_space = dolfin.VectorSpaceBasis([null_vec])
dolfin.as_backend_type(A).set_nullspace(null_space)

null_space.orthogonalize(L)

# solve the linear system
solver.solve(sol.vector(), L)

u_sol, p_sol = sol.split()
u_sol.rename("u", "u")
p_sol.rename("p", "p")

# output the solution
outfile = dolfin.XDMFFile('data/two-phase-u.xdmf')
outfile.write(u_sol)
outfile.close()
outfile = dolfin.XDMFFile('data/two-phase-p.xdmf')
outfile.write(p_sol)
outfile.close()
