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
surface_cell = surface_mesh.ufl_cell()
X_element = dolfin.VectorElement('P', surface_cell, 1, dim=gdim)
X_space = dolfin.FunctionSpace(surface_mesh, X_element)
K_element = dolfin.FiniteElement('P', surface_cell, 1)
K_space = dolfin.FunctionSpace(surface_mesh, K_element)
# XK_element = dolfin.MixedElement([X_element, K_element])
# XK_space = dolfin.FunctionSpace(surface_mesh, XK_element)

# get the surface identity from mesh
x0 = dolfin.Function(X_space)
x0_array = x0.vector().get_local()
v2d = dolfin.vertex_to_dof_map(X_space)
coords = surface_mesh.coordinates()
x0_array[np.ix_(v2d)] = coords.flatten()
x0.vector().set_local(x0_array)
x0.rename("identify", "1")

# get the trial and test functions
# x, kappa = dolfin.TrialFunctions(XK_space)
# eta, chi = dolfin.TestFunctions(XK_space)
x = dolfin.TrialFunction(X_space)
eta = dolfin.TestFunction(X_space)

# the bilinear form for the mean curvature vector
a_2 = dolfin.dot(x, eta)*dx 
l_2 = dolfin.inner(dolfin.grad(x0), dolfin.grad(eta)) * dx

# solve the linear system
# mcv = dolfin.Function(XK_space)[0] # mean curvature vector
mcv = dolfin.Function(X_space)
A = dolfin.assemble(a_2)
L = dolfin.assemble(l_2)
dolfin.solve(A, mcv.vector(), L)

# calculate the initial curvature
kappa0 = dolfin.project(dolfin.Constant(1/2) * dolfin.dot(nu, mcv), K_space)
kappa0.rename("curvature", "2")
nu_h = dolfin.project(nu, X_space)
nu_h.rename("normal", "3")
outfile = dolfin.XDMFFile('data/{}.xdmf'.format(meshPrefix))
outfile.parameters.update({
    "functions_share_mesh": True,
    "rewrite_function_mesh": False
    })
outfile.write(x0, 0.)
outfile.write(nu_h, 0.)
outfile.write(kappa0, 0.)
outfile.close()


None
