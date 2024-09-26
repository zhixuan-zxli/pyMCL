import numpy as np
from scipy import sparse as sp
from scikits.umfpack import spsolve
from matplotlib import pyplot

def u(x: np.ndarray) -> np.ndarray:
    return np.cos(4.0*x)
def du(x: np.ndarray) -> np.ndarray:
    return -4.0 * np.sin(4.0*x)
def ddu(x: np.ndarray) -> np.ndarray:
    return -4.0**2 * np.cos(4.0*x)
def d4u(x: np.ndarray) -> np.ndarray:
    return 4.0**4 * np.cos(4.0*x)

# test Poisson equation
def testFiniteDifference(xi_b: np.ndarray):
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

    # 3. test the Poisson equation
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
    _, ax = pyplot.subplots()
    ax.plot(xi_c[2:-2], uu[1:-1], 'o', label="numeric")
    ax.plot(xi_c[2:-2], u(xi_c[2:-2]), '-', label="reference")
    ax.legend(); ax.set_title("Poisson"); # pyplot.draw()
    err = uu[1:-1] - u(xi_c[2:-2])
    print("Poisson, n = {}, inf-norm error = {:.2e}".format(n, np.linalg.norm(err, ord=np.inf)))
    del val_up, val_lo, val_diag, A, b

    # 4. test the biharmonic equation
    val_up1 = np.zeros((n+3, ))
    val_up1[2:-1] = 24.0 * dd_table[4][:,3]
    val_up2 = np.zeros((n+2, ))
    val_up2[2:] = 24.0 * dd_table[4][:,4]
    val_lo1 = np.zeros((n+3, ))
    val_lo1[1:-2] = 24.0 * dd_table[4][:,1]
    val_lo2 = np.zeros((n+2, ))
    val_lo2[:-2] = 24.0 * dd_table[4][:,0]
    val_diag = np.zeros((n+4, ))
    val_diag[2:-2] = 24.0 * dd_table[4][:,2]
    # set the ghost extrapolation
    val_up3 = np.zeros((n+1, ))
    val_diag[0] = -1/16; val_up1[0] = 9/16; val_up2[0] = 9/16; val_up3[0] = -1/16 # Dirichlet
    val_lo1[0] = 1/(24*h[0]); val_diag[1] = -9/(8*h[0]); val_up1[1] = 9/(8*h[0]); val_up2[1] = -1/(24*h[0]) # Neumann
    val_lo3 = np.zeros((n+1, ))
    val_lo2[-2] = 1/(24*h[1]); val_lo1[-2] = -9/(8*h[1]); val_diag[-2] = 9/(8*h[1]); val_up1[-1] = -1/(24*h[1]) # Neumann
    val_lo3[-1] = -1/16; val_lo2[-1] = 9/16; val_lo1[-1] = 9/16; val_diag[-1] = -1/16 # Dirichlet
    # prepare the linear system
    A = sp.diags((val_diag, val_up1, val_up2, val_up3, val_lo1, val_lo2, val_lo3), 
                 (0, 1, 2, 3, -1, -2, -3), (n+4, n+4), "csr")
    b = np.zeros((n+4, ))
    b[2:-2] = d4u(xi_c[2:-2])
    b[0] = u(xi_b[0]); b[1] = du(xi_b[0])
    b[-2] = du(xi_b[-1]); b[-1] = u(xi_b[-1])
    # solve the linear system
    uu = spsolve(A, b)
    _, ax = pyplot.subplots()
    ax.plot(xi_c[2:-2], uu[2:-2], 'o', label="numeric")
    ax.plot(xi_c[2:-2], u(xi_c[2:-2]), '-', label="reference")
    ax.legend(); ax.set_title("Biharmonic"); #pyplot.draw()
    err = uu[2:-2] - u(xi_c[2:-2])
    print("Biharmonic, n = {}, inf-norm error = {:.2e}".format(n, np.linalg.norm(err, ord=np.inf)))
    del val_up1, val_up2, val_up3, val_lo1, val_lo2, val_lo3, val_diag, A, b



if __name__ == "__main__":
    # xi_b = np.linspace(0.0, 1.0, 33)
    xi_b = np.concatenate((
        np.linspace(0.0, 0.25, 33), 
        np.linspace(0.25, 0.5, 17)[1:], 
        np.linspace(0.5, 1.0, 9)[1:]
    ))
    #
    testFiniteDifference(xi_b)
    pyplot.show()
