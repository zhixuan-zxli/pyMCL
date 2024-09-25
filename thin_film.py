import numpy as np
from scipy import optimize as opt
from scipy import sparse as sp
from scikits.umfpack import spsolve
from matplotlib import pyplot

xi_range = None

# the mesh mapping
def varphi(xi: np.ndarray, mesh_params: tuple[float]) -> tuple[np.ndarray]:
    # x (?,) query points in [-1, 1]
    # params = (a,b)
    # return: (varphi, J, J', J'')
    global xi_range
    a, b = mesh_params

    f = lambda x: x * (x**2+b)
    df = lambda x: 3*x**2 + b
    ddf = lambda x: 6*x
    dddf = lambda x: 6.0
    f_range = (-1.0, 3.0)
    if xi_range is None:
        xi_range = [opt.root_scalar(lambda x: f(x) - a, method="toms748", bracket=(-10.0, 10.0), fprime=df).root for a in f_range]

    y = xi_range[0] * (1-xi) + xi_range[1] * xi
    return (
        (f(y) - f_range[0]) / (f_range[1] - f_range[0]) * (a*4), 
        df(y) * (xi_range[1] - xi_range[0]) / (f_range[1] - f_range[0]) * (a*4), 
        ddf(y) * (xi_range[1] - xi_range[0]) ** 2 / (f_range[1] - f_range[0]) * (a*4), 
        dddf(y) * (xi_range[1] - xi_range[0]) ** 3 / (f_range[1] - f_range[0]) * (a*4)
    )

# test Poisson equation
def testPoisson(xi_c: np.ndarray, xi_b: np.ndarray, dxi: float, mesh_params):
    def exact_sol(x: np.ndarray) -> np.ndarray:
        return np.cos(np.pi*x)
    def rhs(x: np.ndarray) -> np.ndarray:
        return np.pi**2 * np.cos(np.pi*x)
    n = xi_c.size # number of cells
    x, J, Jp, _ = varphi(xi_c, mesh_params) # (n, ), (n, )
    # set up the Laplacian
    val_diag = np.zeros((n+2, ))
    val_diag[1:-1] = 2.0 / (dxi * J)**2
    val_up = np.zeros((n+1, ))
    val_up[1:] = (Jp/(2.0*dxi*J) - 1.0/dxi**2) / J**2
    val_lo = np.zeros((n+1, ))
    val_lo[:-1] = (-Jp/(2.0*dxi*J) - 1.0/dxi**2) / J**2
    # set up the ghost points for Dirichlet condition
    val_diag[0] = 1.0/2; val_up[0] = 1.0/2
    val_diag[-1] = 1.0/2; val_lo[-1] = 1.0/2
    # assemble the matrix
    A = sp.diags((val_diag, val_up, val_lo), (0, 1, -1), (n+2, n+2), "csr")
    # prepare the rhs
    b = np.zeros((n+2, ))
    b[1:-1] = rhs(x)
    x_b, _, _, _ = varphi(xi_b, mesh_params)
    b[0] = exact_sol(x_b[0]); b[-1] = exact_sol(x_b[-1])
    # solve the linear system
    uu = spsolve(A, b)
    # plot the solution
    pyplot.plot(x, uu[1:-1], 'o', label="numeric")
    pyplot.plot(x, exact_sol(x), '-', label="reference")
    err = uu[1:-1] - exact_sol(x)
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
    N = 32 # number of cells for the wet part
    dxi = 0.5 / N
    a, b = 0.25, 1e-2
    all_xi = np.linspace(0.0, 1.0, 4*N+1)
    xi_c = all_xi[1::2] # the volume center
    xi_b = all_xi[0::2] # the volume boundary
    #
    pyplot.figure()
    testPoisson(xi_c, xi_b, dxi, (a, b))
    # testBiharmonic(xi_c, xi_b, delta_xi, (A,b))
    pyplot.legend()
    pyplot.show()
