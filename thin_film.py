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
    xi_c = (xi_g[:-1] + xi_g[1:]) / 2 # get the cell centers (with ghosts)

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

    # initial value
    slip = 1e-4 # the slip length
    a = 1.0 # the CL position
    adot = 1e1
    # adot = 0.0
    h = np.zeros((n+4, ))
    h[2:-2] = 1 - np.exp(5.0*(xi_c[2:-2]-1))
    h[2:-2] *= 0.4 + 0.05 * np.cos(20*xi_c[2:-2])

    pyplot.ion()
    _, ax = pyplot.subplots()
    ax.plot(xi_c[2:-2], h[2:-2], '-')
    ax.set_ylim(-0.1, 0.5); ax.axis("equal")

    dt = 1.0/(1024*1024)
    Ca = 1.0
    maxStep = 1024
    for m in range(maxStep):
        a = 1.0 + m * dt * adot
        # fill the ghosts near the symmetric boundary at x=0 and at x=a
        h[0], h[1] = h[3], h[2]
        h[-2] = -h[-3]
        # interpolate to cell boundaries
        h_mid = (h[3:-2] + h[2:-3]) / 2 # (n-1, )
        # calculate the flux coefficients at the cell boundaries
        fc = np.concatenate(((0.0,), h_mid**2 * (h_mid/3 + slip), (0.0,))) # (n+1, ), zero flux at both boundaries
        # build the FD scheme
        fdtab = np.zeros((n, 5))
        fdtab[:,0] = - fc[:-1] * dd_table[3][:-1, 0] # (n, )
        fdtab[:,4] = fc[1:] * dd_table[3][1:, 3]    # (n, )
        for j in range(1, 4):
            fdtab[:,j] = fc[1:] * dd_table[3][1:,j-1] - fc[:-1] * dd_table[3][:-1,j] # (n, )
        fdtab = 6 * fdtab / (xi_b[1:] - xi_b[:-1])[:,np.newaxis]
        # assemble the sparse matrix; dispose the rightmost ghost cell
        v_diag = np.zeros((n+3, ))
        v_diag[2:-1] = fdtab[:,2]
        v_up1 = np.zeros((n+2, ))
        v_up1[2:] = fdtab[:,3]
        v_lo1 = np.zeros((n+2, ))
        v_lo1[1:-1] = fdtab[:,1]
        v_up2 = np.zeros((n+1, ))
        v_up2[2:] = fdtab[:-1,4]
        v_lo2 = np.zeros((n+1, ))
        v_lo2[:-1] = fdtab[:,0]
        LL = sp.diags((v_diag, v_up1, v_up2, v_lo1, v_lo2), (0, 1, 2, -1, -2), (n+3, n+3), "csr")
        # assemble the boundary conditions
        v_diag[:] = 0.0; v_up1[:] = 0.0; v_up2[:] = 0.0; v_lo1[:] = 0.0; v_lo2[:] = 0.0
        v_up1[0] = -1.0; v_up2[0] = 1.0 # symmetry
        v_lo1[0] = -1.0; v_up2[1] = 1.0 # symmetry
        v_lo1[-1] = 0.5; v_diag[-1] = 0.5 # zero boundary condition
        G = sp.diags((v_diag, v_up1, v_up2, v_lo1, v_lo2), (0, 1, 2, -1, -2), (n+3, n+3), "csr")
        v_diag[:] = 0.0; v_diag[2:-1] = 1.0
        I = sp.diags((v_diag, ), (0, ), (n+3, n+3), "csr")
        del v_diag, v_up1, v_up2, v_lo1, v_lo2
        # calculate the advection term using upwind
        adv = np.zeros((n, ))
        adv[:-1] = (h[3:-2] - h[2:-3]) / (xi_c[3:-2] - xi_c[2:-3]) # (n-1, )
        adv[-1] = -2*h[-3] / (xi_b[-1] - xi_b[-2])
        adv *= xi_c[2:-2] * adot / a
        # assemble the linear system
        A = I + (dt / (Ca * a**4)) * LL + G
        b = np.zeros((n+3, ))
        b[2:-1] = h[2:-2] + dt * adv
        h_next = spsolve(A, b)
        diff = h_next[2:-1] - h[2:-2]
        h[:-1] = h_next
        vol = np.sum(h_next[2:-1] * a * (xi_b[1:] - xi_b[:-1]))
        assert np.all(h[2:-2] >= 0.)
        #
        ax.clear()
        ax.plot(a * xi_c[2:-2], h[2:-2], '-')
        print("t = {:.6f}, step = {}/{}, diff = {:.2e}, vol = {:.4e}, a = {:.5f}".format((m+1)*dt, m, maxStep, np.linalg.norm(diff, np.inf), vol, a)); 
        ax.set_xlim(0.0, 1.5); ax.set_ylim(-0.1, 0.5); # ax.axis("equal")
        pyplot.draw(); pyplot.pause(1e-2)
        pass

    pyplot.ioff()
    print("Finished.")

    if False:
        # 3. build the negative Laplacian operator
        val_up = np.zeros((n+3, ))
        val_up[2:-1] = -2 * dd_table[2][1:-1,2]
        val_lo = np.zeros((n+3, ))
        val_lo[1:-2] = -2 * dd_table[2][1:-1,0]
        val_diag = np.zeros((n+4, ))
        val_diag[2:-2] = -2 * dd_table[2][1:-1,1]
        # assemble the matrix
        L = sp.diags((val_diag, val_up, val_lo), (0, 1, -1), (n+4, n+4), "csr")
        del val_up, val_lo, val_diag

        # 4. build the biharmonic operator
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
        A = sp.diags((val_diag, val_up1, val_up2, val_lo1, val_lo2), (0, 1, 2, -1, -2), (n+4, n+4), "csr")

        # set the ghost extrapolation
        val_up1[:] = 0.0; val_up2[:] = 0.0; val_diag[:] = 0.0; val_lo1[:] = 0.0; val_lo2[:] = 0.0
        val_up3 = np.zeros((n+1, ))
        val_diag[0] = -1/16; val_up1[0] = 9/16; val_up2[0] = 9/16; val_up3[0] = -1/16 # Dirichlet
        # val_lo1[0] = 1/(24*h[0]); val_diag[1] = -9/(8*h[0]); val_up1[1] = 9/(8*h[0]); val_up2[1] = -1/(24*h[0]) # Neumann
        val_lo1[0] = 1.0; val_diag[1] = -1.0; val_up1[1] = -1.0; val_up2[1] = 1.0; # second-order
        val_lo3 = np.zeros((n+1, ))
        val_lo2[-2] = 1/(24*h[1]); val_lo1[-2] = -9/(8*h[1]); val_diag[-2] = 9/(8*h[1]); val_up1[-1] = -1/(24*h[1]) # Neumann
        val_lo3[-1] = -1/h[1]**3; val_lo2[-1] = 3/h[1]**3; val_lo1[-1] = -3/h[1]**3; val_diag[-1] = 1/h[1]**3 # third-order derivative
        G = sp.diags((val_diag, val_up1, val_up2, val_up3, val_lo1, val_lo2, val_lo3), 
                    (0, 1, 2, 3, -1, -2, -3), (n+4, n+4), "csr")
        del val_up1, val_up2, val_up3, val_lo1, val_lo2, val_lo3, val_diag

        # account for the jump
        c = np.where(xi_c >= 0.5, 0.5 * 1.0/6*(xi_c - 0.5)**3, 0.0)
        Ac = A @ c

        # prepare the linear system
        bm = 1e-5 # the bending modulus
        b = np.zeros((n+4, ))
        b[2:-2] = np.where(xi_c[2:-2] >= 0.5, -1.0, 0.0)
        # b[0] = u(xi_b[0]); b[1] = du(xi_b[0])
        # b[-2] = du(xi_b[-1]); b[-1] = u(xi_b[-1])
        # solve the linear system
        uu = spsolve(bm * A + L + G, b + Ac)
        _, ax = pyplot.subplots()
        ax.plot(xi_c[2:-2], uu[2:-2], '-', label="numeric")
        # ax.plot(xi_c[2:-2], u(xi_c[2:-2]), '-', label="reference")
        ax.legend(); ax.axis("equal"); ax.set_title("Plate bending with tension"); #pyplot.draw()
        # err = uu[2:-2] - u(xi_c[2:-2])
        # print("Biharmonic, n = {}, inf-norm error = {:.2e}".format(n, np.linalg.norm(err, ord=np.inf)))



if __name__ == "__main__":
    # xi_b = np.linspace(0.0, 1.0, 65)
    xi_b = np.concatenate((
        np.linspace(0.0, 0.5, 33), 
        np.linspace(1/2, 3/4, 33)[1:], 
        np.linspace(3/4, 7/8, 33)[1:],
        np.linspace(7/8, 15/16, 33)[1:],
        np.linspace(15/16, 1.0, 65)[1:],
    ))
    #
    testFiniteDifference(xi_b)
    pyplot.show()
