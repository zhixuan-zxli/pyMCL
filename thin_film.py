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
    # 1. set up the grid. 
    n = xi_b.size - 1 # number of cells
    h = xi_b[1] - xi_b[0], xi_b[-1] - xi_b[-2]
    xi_g = np.concatenate(((xi_b[0] - 2*h[0], xi_b[0] - h[0]), xi_b, (xi_b[-1] + h[1], xi_b[-1] + 2*h[1]))) # add ghost points
    xi_c = (xi_g[:-1] + xi_g[1:]) / 2 # get the cell centers
    # 2. build the divided difference table. 
    dd_table = [None]
    # first-order difference
    dd1 = np.zeros((n+3, 2))
    dd1[:,1] = 1/(xi_c[1:] - xi_c[:-1])
    dd1[:,0] = -dd1[:,1]
    dd_table.append(dd1)
    for k in range(2, 5): # build the k-th order difference table
        ddk = np.zeros((n+4-k, k+1))
        dd_last = dd_table[-1]
        ddk[:,0] = -dd_last[:-1,0]
        ddk[:,-1] = dd_last[1:,-1]
        for j in range(1, k):
            ddk[:,j] = dd_last[1:,j-1] - dd_last[:-1, j]
        ddk /= (xi_c[k:] - xi_c[:-k])[:,np.newaxis]
        dd_table.append(ddk)
    # set up the Laplacian
    val_up = np.zeros((n+1, ))
    val_up[1:] = -2 * dd_table[2][1:-1,2]
    val_lo = np.zeros((n+1, ))
    val_lo[:-1] = -2 * dd_table[2][1:-1,0]
    val_diag = np.zeros((n+2, ))
    val_diag[1:-1] = -2 * dd_table[2][1:-1,1]
    # set up the Dirichlet condition
    val_diag[0] = 1.0/2; val_up[0] = 1.0/2
    val_diag[-1] = 1.0/2; val_lo[-1] = 1.0/2
    # assemble the matrix
    A = sp.diags((val_diag, val_up, val_lo), (0, 1, -1), (n+2, n+2), "csr")
    # prepare the rhs
    b = np.zeros((n+2, ))
    b[1:-1] = -ddu(xi_c[2:-2])
    b[0] = u(xi_b[0]); b[-1] = u(xi_b[-1])
    # solve the linear system
    uu = spsolve(A, b)
    # plot the solution
    pyplot.plot(xi_c[2:-2], uu[1:-1], 'o', label="numeric")
    pyplot.plot(xi_c[2:-2], u(xi_c[2:-2]), '-', label="reference")
    err = uu[1:-1] - u(xi_c[2:-2])
    print("n = {}, inf-norm error = {:.2e}".format(n, np.linalg.norm(err, ord=np.inf)))


if __name__ == "__main__":
    # xi_b = np.linspace(0.0, 1.0, 33)
    xi_b = np.concatenate((
        np.linspace(0.0, 0.25, 33), 
        np.linspace(0.25, 0.5, 17)[1:], 
        np.linspace(0.5, 1.0, 9)[1:]
    ))
    #
    pyplot.figure()
    testPoisson(xi_b)
    # testBiharmonic(xi_c, xi_b, delta_xi, (A,b))
    pyplot.legend()
    pyplot.show()
