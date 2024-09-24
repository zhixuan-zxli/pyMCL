import numpy as np
from scipy import sparse as sp
from scikits.umfpack import spsolve
from matplotlib import pyplot

# the mesh mapping
def varphi(xi: np.ndarray, params: tuple[float]) -> tuple[np.ndarray]:
    # x (?,) query points in [-1, 1]
    # params = (a,b)
    # return: (varphi, varphi', varphi'')
    a, b = params

    # this parabolic function maps [0,1] onto itself with varphi1(1/2) = a
    def varphi1(x: np.ndarray) -> tuple[np.ndarray]:
        return x * (2*(1-2*a) * x + 4*a - 1), 4*(1-2*a)*x + 4*a-1, 4*(1-2*a)
    
    # this cubic maps [-1, 1] onto itself with slope b near x=0
    def varphi2(x: np.ndarray) -> tuple[np.ndarray]:
        return x * ((1-b)*x**2 + b), 3*(1-b)*x**2+b, 6*(1-b)*x
    
    r2 = varphi2(xi)
    r1 = varphi1((r2[0]+1)/2)
    return (r1[0], 0.5 * r1[1] * r2[1], 0.25 * r1[2] * r2[1]**2 + 0.5 * r1[1] * r2[2])

# test Poisson equation
def testPoisson(xi_c: np.ndarray, dxi: float, mesh_params):
    def exact_sol(x: np.ndarray) -> np.ndarray:
        return np.cos(np.pi*x)
    def rhs(x: np.ndarray) -> np.ndarray:
        return np.pi**2 * np.cos(np.pi*x)
    n = xi_c.size # number of cells
    x, J, Jp = varphi(xi_c, mesh_params) # (n, ), (n, )
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
    b[0] = exact_sol(0.); b[-1] = exact_sol(1.)
    # solve the linear system
    u = spsolve(A, b)
    # plot the solution
    pyplot.plot(x, u[1:-1], 'o', label="numeric")
    pyplot.plot(x, exact_sol(x), '-', label="reference")


if __name__ == "__main__":
    A = 1.0/2
    b = 1e-2
    N = 32 # number of volumes for the wet part
    delta_xi = 1.0 / N
    all_xi = np.linspace(-1.0, 1.0, 4*N+1)
    xi_c = all_xi[1::2] # the volume center
    xi_b = all_xi[0::2] # the volume boundary
    #
    pyplot.figure()
    testPoisson(xi_c, delta_xi, (A, b))
    pyplot.legend()
    pyplot.show()
