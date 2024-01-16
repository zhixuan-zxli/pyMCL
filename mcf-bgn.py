import numpy as np
import dolfin
from msh2xdmf import import_mesh
from matplotlib import pyplot

tdim = 2 # dimension of the manifold
gdim = 3 # dimension of the ambient space
meshPrefix = 'sphere'

# first import mesh
surface_mesh, boundary_marker, physical_table = import_mesh(prefix=meshPrefix, tdim=tdim, gdim=gdim, directory='mesh')

# fix the mesh normal
surface_mesh.init_cell_orientations(dolfin.Expression(("x[0]", "x[1]", "x[2]"), degree=1))
# save the mesh normal
# DG_element = dolfin.VectorElement('DG', surface_mesh.ufl_cell(), 1, dim = gdim)
# nh = dolfin.project(n, dolfin.FunctionSpace(surface_mesh, DG_element))
# outfile = dolfin.XDMFFile('data/sphere.xdmf')
# outfile.write(nh)
# outfile.close()

# get the normal
nu = dolfin.CellNormal(surface_mesh)
dx = dolfin.Measure('dx', surface_mesh)

m = dolfin.assemble(dolfin.dot(nu,nu) * dx)
print('Initial surface measure = {}'.format(m))

# Define the function space
X_element = dolfin.VectorElement('P', dolfin.triangle, 1, dim=gdim)
K_element = dolfin.FiniteElement('P', dolfin.triangle, 1)
XK_element = dolfin.MixedElement([X_element, K_element])
XK_space = dolfin.FunctionSpace(surface_mesh, XK_element)

# get the trial and test functions
x, kappa = dolfin.TrialFunction(XK_space)
eta, chi = dolfin.TestFunctions(XK_space)

# the bilinear form for the mean curvature vector
a_2 = dolfin.dot(x, eta) + dolfin.inner(dolfin.grad(x), dolfin.grad(eta))
rhs = dolfin.dot(dolfin.Constant((0, 0)), eta)

# solve the linear system
mcv = dolfin.Function(XK_space)[0] # mean curvature vector
A = dolfin.assemble(a_2)
RHS = dolfin.assemble(rhs)
dolfin.solve(A, mcv.vector(), RHS)

# calculate the initial curvature
kappa0 = dolfin.Function(XK_space)[1]
mcv_array = mcv.vector().get_local()
kappa0.vector().set_local(np.sqrt(mcv_array.sum(axis=1)))
outfile = dolfin.XDMFFile('data/{}'.format(meshPrefix))
outfile.write(kappa0)
outfile.close()


None
