import numpy as np
import dolfin
from msh2xdmf import import_mesh
from matplotlib import pyplot

# load the mesh
mesh, boundary_marker, subdomain_marker, physical_table = import_mesh(prefix='two-phase', subdomains=True, tdim=2, gdim=2, directory="mesh")

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
P1 = dolfin.FiniteElement('CG', cell, degree=1)
P0 = dolfin.FiniteElement('DG', cell, degree=0)
AugTHSpace = dolfin.FunctionSpace(mesh, dolfin.MixedElement([U, P1, P0]), constrained_domain = periodic_bc)

# define the flow boundary conditions
noSlipBC_T = dolfin.DirichletBC(AugTHSpace.sub(0), dolfin.Constant((0.0, 0.0)), \
                              boundary_marker, physical_table["top"])
noPen_L = dolfin.DirichletBC(AugTHSpace.sub(0).sub(1), dolfin.Constant(0.0), \
                                  boundary_marker, physical_table["dryleft"])
noPen_W = dolfin.DirichletBC(AugTHSpace.sub(0).sub(1), dolfin.Constant(0.0), \
                                  boundary_marker, physical_table["wet"])
noPen_R = dolfin.DirichletBC(AugTHSpace.sub(0).sub(1), dolfin.Constant(0.0), \
                                  boundary_marker, physical_table["dryright"])
flow_bcs = [noSlipBC_T, noPen_L, noPen_W, noPen_R]

# define the physical parameters
phys = {"gamma":1.0, "kappa":-0.2}

# define the variational problem
u, p1, p0 = dolfin.TrialFunctions(AugTHSpace)
v, q1, q0 = dolfin.TestFunctions(AugTHSpace)
# a = dolfin.inner(dolfin.grad(u), dolfin.grad(v)) * dx + dolfin.div(v) * p * dx + dolfin.div(u) * q * dx
a = ( dolfin.inner(dolfin.grad(u), dolfin.grad(v)) + dolfin.div(v) * (p1+p0) + dolfin.div(u) * (q1+q0) ) * dx
l = dolfin.Constant(phys["gamma"] * phys["kappa"]) * dolfin.dot(n('+'), v('+')) * dS # \
    # + dolfin.dot(dolfin.Constant((0.0, 0.0)), v) * dx # a hack to fix the interior facet normal
b = dolfin.inner(dolfin.grad(u), dolfin.grad(v))*dx + (p1+p0)*(q1+q0)*dx # the preconditioning matrix

# assemble the linear system
A, L = dolfin.assemble_system(a, l, flow_bcs)
B, _ = dolfin.assemble_system(b, l, flow_bcs)

# set up the solver
solver = dolfin.KrylovSolver("gmres", "amg")
solver.set_operators(A, B)
solver.parameters["report"] = True

# create the null vector
sol = dolfin.Function(AugTHSpace)
nv1 = dolfin.Vector(sol.vector())
nv0 = dolfin.Vector(sol.vector())
AugTHSpace.sub(1).dofmap().set(nv1, 1.0)
AugTHSpace.sub(2).dofmap().set(nv0, 1.0)
nv1 *= 1.0 / nv1.norm("l2")
nv0 *= 1.0 / nv0.norm("l2")

# attach the null space to the PETSc matrix
null_space = dolfin.VectorSpaceBasis([nv1, nv0])
dolfin.as_backend_type(A).set_nullspace(null_space)

null_space.orthogonalize(L)

# solve the linear system
solver.solve(sol.vector(), L)

u_sol, p1_sol, p0_sol = sol.split()
u_sol.rename("u", "u")

# Convert the P1+P0 pressure to a DG1 function
PP = dolfin.FunctionSpace(mesh, dolfin.FiniteElement('DG', cell, degree=1), constrained_domain = periodic_bc)
pp1 = dolfin.project(p1_sol, PP)
pp0 = dolfin.project(p0_sol, PP)
pp1.vector().axpy(1.0, pp0.vector())
pp1.rename("p", "p")

# output the solution
outfile = dolfin.XDMFFile('data/two-phase-u.xdmf')
outfile.write(u_sol)
outfile.close()
outfile = dolfin.XDMFFile('data/two-phase-p.xdmf')
outfile.write(pp1)
outfile.close()
