import numpy as np
from scipy import sparse as sp
from scikits.umfpack import spsolve
from matplotlib import pyplot

def u(x: np.ndarray) -> np.ndarray:
    return np.cos(np.pi*x)
def ddu(x: np.ndarray) -> np.ndarray:
    return -np.pi**2 * np.cos(np.pi*x)

# test Poisson equation
def testPoisson(xi_b: np.ndarray):
    n = xi_b.size - 1 # number of cells
    # add ghost points
    h = xi_b[1] - xi_b[0], xi_b[-1] - xi_b[-2]
    xi_g = np.concatenate(((xi_b[0] - h[0], ), xi_b, (xi_b[-1] + h[1],)))
    # get the cell centers
    xi_c = (xi_g[:-1] + xi_g[1:]) / 2
    # # set up the Laplacian
    val_up = np.zeros((n+1, ))
    val_up[1:] = -2/((xi_c[2:] - xi_c[:-2]) * (xi_c[2:] - xi_c[1:-1]))
    val_lo = np.zeros((n+1, ))
    val_lo[:-1] = -2/((xi_c[2:] - xi_c[:-2]) * (xi_c[1:-1] - xi_c[:-2]))
    val_diag = np.zeros((n+2, ))
    val_diag[1:-1] = -val_up[1:] - val_lo[:-1]
    # set up the Dirichlet condition
    val_diag[0] = 1.0/2; val_up[0] = 1.0/2
    val_diag[-1] = 1.0/2; val_lo[-1] = 1.0/2
    # assemble the matrix
    A = sp.diags((val_diag, val_up, val_lo), (0, 1, -1), (n+2, n+2), "csr")
    # prepare the rhs
    b = np.zeros((n+2, ))
    b[1:-1] = -ddu(xi_c[1:-1])
    b[0] = u(xi_b[0]); b[-1] = u(xi_b[-1])
    # solve the linear system
    uu = spsolve(A, b)
    # plot the solution
    pyplot.plot(xi_c[1:-1], uu[1:-1], 'o', label="numeric")
    pyplot.plot(xi_c[1:-1], u(xi_c[1:-1]), '-', label="reference")
    err = uu[1:-1] - u(xi_c[1:-1])
    print("n = {}, inf-norm error = {:.2e}".format(n, np.linalg.norm(err, ord=np.inf)))

# test biharmonic equation
# def testBiharmonic(xi_c: np.ndarray, xi_b: np.ndarray, dxi: float, mesh_params):
#     def u(x: np.ndarray) -> np.ndarray:
#         return np.cos(np.pi*x)
#     def du(x: np.ndarray) -> np.ndarray:
#         return -np.pi * np.sin(np.pi*x)
#     def rhs(x: np.ndarray) -> np.ndarray:
#         return np.pi**4 * np.cos(np.pi*x)
#     n = xi_c.size # number of cells
#     x, J, Jp, Jpp, Jppp = varphi(xi_c, mesh_params) # (n, ), (n, )
#     # set up the interior finite difference
#     A4 = 1 / J**4
#     A3 = -6*Jp/J**5
#     A2 = (-4*Jpp + 15*Jp**2/J)/J**5
#     A1 = (-Jppp + 10*Jp*Jpp/J - 15*Jp**3/J**2)/J**5
#     val_diag = np.zeros((n+4,))
#     val_diag[2:-2] = 6*A4/dxi**4 - 2*A2/dxi**2
#     val_up1 = np.zeros((n+3,))
#     val_up1[2:-1] = -4*A4/dxi**4 - A3/dxi**3 + A2/dxi**2 + A1/(2*dxi)
#     val_lo1 = np.zeros((n+3,))
#     val_lo1[1:-2] = -4*A4/dxi**4 + A3/dxi**3 + A2/dxi**2 - A1/(2*dxi)
#     val_up2 = np.zeros((n+2,))
#     val_up2[2:] = A4/dxi**4 + A3/(2*dxi**3)
#     val_lo2 = np.zeros((n+2,))
#     val_lo2[:-2] = A4/dxi**4 - A3/(2*dxi**3)
#     # set up the boundary conditions
#     _, J, _, _, _ = varphi(xi_b, mesh_params)
#     val_up1[0] = 1.0/2; val_up2[0] = 1.0/2 # Dirichlet
#     val_lo1[0] = 1/(24*dxi*J[0]); val_diag[1] = -9/(8*dxi*J[0]); val_up1[1] = 9/(8*dxi*J[0]); val_up2[1] = -1/(24*dxi*J[0]) # Neumann
#     val_lo2[-2] = 1/(24*dxi*J[-1]); val_lo1[-2] = -9/(8*dxi*J[-1]); val_diag[-2] = 9/(8*dxi*J[-1]); val_up1[-1] = -1/(24*dxi*J[-1]) # Neumann
#     val_lo2[-1] = 1.0/2; val_lo1[-1] = 1.0/2 # Dirichlet
#     A = sp.diags((val_diag, val_up1, val_up2, val_lo1, val_lo2), (0, 1, 2, -1, -2), (n+4, n+4), "csr")
#     # prepare the rhs
#     b = np.zeros((n+4, ))
#     b[2:-2] = rhs(x)
#     b[0] = u(x[0]); b[1] = du(x[0])
#     b[-2] = du(x[-1]); b[-1] = u(x[-1])    
#     # plot the solution
#     uu = spsolve(A, b)
#     pyplot.plot(x, uu[2:-2], 'o', label="numeric")
#     pyplot.plot(x, u(x), '-', label="reference")


if __name__ == "__main__":
    xi_b = np.linspace(0.0, 1.0, 33)
    # xi_b = np.concatenate((
    #     np.linspace(0.0, 0.25, 33), 
    #     np.linspace(0.25, 0.5, 17)[1:], 
    #     np.linspace(0.5, 1.0, 9)[1:]
    # ))
    #
    pyplot.figure()
    testPoisson(xi_b)
    # testBiharmonic(xi_c, xi_b, delta_xi, (A,b))
    pyplot.legend()
    pyplot.show()
