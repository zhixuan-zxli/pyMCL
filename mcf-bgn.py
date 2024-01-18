import numpy as np
import dolfin
from msh2xdmf import import_mesh
from matplotlib import pyplot
from getSubspaceAndIndex import *

tdim = 2 # dimension of the manifold
gdim = tdim+1 # dimension of the ambient space
meshPrefix = 'dumbbell'

outfile = dolfin.XDMFFile('data/{}.xdmf'.format(meshPrefix))
# outfile.parameters.update({
#     "functions_share_mesh": True,
#     "rewrite_function_mesh": False
#     })

# first import mesh
surface_mesh, boundary_marker, physical_table = import_mesh(prefix=meshPrefix, tdim=tdim, gdim=gdim, directory='mesh')

# fix the mesh normal
surface_mesh.init_cell_orientations(dolfin.Expression(("x[0]", "x[1]", "x[2]"), degree=1))

# get the symbols
nu = dolfin.CellNormal(surface_mesh)
dx = dolfin.Measure('dx', surface_mesh)
# x = dolfin.SpatialCoordinate(surface_mesh)

# Define the function space
surface_cell = surface_mesh.ufl_cell()
X_element = dolfin.VectorElement('P', surface_cell, 1, dim=gdim)
K_element = dolfin.FiniteElement('P', surface_cell, 1)
XK_space = dolfin.FunctionSpace(surface_mesh, dolfin.MixedElement([X_element, K_element]))
# Get the subspace by collapsing
X_space, X_idx = getSubspaceAndIndex(XK_space, 0)
K_space, K_idx = getSubspaceAndIndex(XK_space, 1)

# output the normal
nu_h = dolfin.project(nu, X_space)
nu_h.rename("nu", "0")
outfile.write(nu_h, 0.0)

# set the initial value: identity of mesh
x_expr = dolfin.Expression(("x[0]", "x[1]"), degree=1) if gdim == 2 else dolfin.Expression(("x[0]", "x[1]", "x[2]"), degree=1)
x_n = dolfin.interpolate(x_expr, X_space)
x_n.rename("x", "1")
k_n = dolfin.Function(K_space)
k_n.rename("kappa", "2")

# get the trial and test functions
x, kappa = dolfin.TrialFunctions(XK_space)
eta, chi = dolfin.TestFunctions(XK_space)

param = {"dt":1.0/1000, "maxStep":50}

# set the linear form for the BGN method
a = dolfin.Constant(1.0/param["dt"]) * dolfin.dot(x, nu) * chi * dx - kappa * chi * dx \
  + kappa * dolfin.dot(nu, eta) * dx + dolfin.inner(dolfin.grad(x), dolfin.grad(eta)) * dx
l = dolfin.Constant(1.0/param["dt"]) * dolfin.dot(x_n, nu) * chi * dx

# solve the linear system
t = 0.0
xk_next = dolfin.Function(XK_space)
x_next = dolfin.Function(X_space)
k_next = dolfin.Function(K_space)
for step in range(param["maxStep"]+1):
    print("step = {0}, t = {1:.3f}, ".format(step, t), end = "")
    # output
    outfile.write(x_n, t)
    outfile.write(k_n, t)
    if step == param["maxStep"]:
        break
    # advance in time
    A = dolfin.assemble(a)
    L = dolfin.assemble(l)
    dolfin.solve(A, xk_next.vector(), L)
    # dolfin.solve(a == l, xk_next)
    # move the mesh
    x_next.vector().set_local(xk_next.vector().get_local()[X_idx[:,1]])
    k_next.vector().set_local(xk_next.vector().get_local()[K_idx[:,1]])
    disp = dolfin.Function(X_space)
    disp.assign(x_next)
    disp.vector().axpy(-1.0, x_n.vector())
    disp_norm = np.abs(disp.vector().get_local()).max()
    print('area = {0}, disp = {1:.2e}'.format(dolfin.assemble(dolfin.Constant(1.0) * dx), disp_norm.item()))
    dolfin.ALE.move(surface_mesh, disp)
    # update the variables
    x_n.assign(x_next)
    k_n.assign(k_next)
    t += param["dt"]

outfile.close()
