import numpy as np
from scipy import sparse as sp
from scikits.umfpack import spsolve
from matplotlib import pyplot
from runner import *

# test Poisson equation
# def testFiniteDifference(xi_b: np.ndarray):

@dataclass
class PhysicalParameters:
    gamma: tuple[float] = (5.0, 5.0, 1.0) # the (effective) surface tension for the wet, dry and the interface
    # Ca: float = gamma[-1]
    slip: float = 1e-4   # the slip length
    theta_Y: float = 0.2
    mu_cl: float = 5.0
    bm: float = 1e-2     # the bending modulus


if __name__ == "__main__":

    solp = SolverParameters(dt = 1/(1024*16), Te=1.0)
    phyp = PhysicalParameters()
    
    # 1. set up the grid. 
    # m = 64
    # xi_b_f = np.concatenate((
    #     np.linspace(0.0, 0.5, m+1), 
    #     np.linspace(1/2, 3/4, m+1)[1:], 
    #     np.linspace(3/4, 7/8, m+1)[1:],
    #     np.linspace(7/8, 15/16, m+1)[1:],
    #     np.linspace(15/16, 31/32, m+1)[1:],
    #     np.linspace(31/32, 63/64, m+1)[1:],
    #     np.linspace(63/64, 1.0, 2*m+1)[1:],
    # ))
    xi_b_f = np.linspace(0.0, 1.0, 1025)
    xi_b = np.concatenate((xi_b_f, 2.0 - xi_b_f[-2::-1])) # cell boundaries
    n_fluid = xi_b_f.size - 1 # number of cells for the fluid (excluding ghosts)
    n_total = xi_b.size - 1   # number of cells total (excluding ghosts)
    
    dxi = xi_b[1:] - xi_b[:-1] # mesh step size on the reference domain
    xi_g = np.concatenate(((xi_b[0] - 2*dxi[0], xi_b[0] - dxi[0]), xi_b, (xi_b[-1] + dxi[-1], xi_b[-1] + 2*dxi[-1]))) # add ghost points
    xi_c = (xi_g[:-1] + xi_g[1:]) / 2 # get the cell centers (with ghosts)
    xi_c_fluid = xi_c[:n_fluid+3]     # the fluid cells, with ghosts
    dxi_at_cl = xi_c[n_fluid+2] - xi_c[n_fluid+1]
    
    # 2. build the divided difference table. 
    dd_table = [None]
    # first-order difference
    dd_1 = np.zeros((n_total+3, 2))
    dd_1[:,1] = 1/(xi_c[1:] - xi_c[:-1])
    dd_1[:,0] = -dd_1[:,1]
    dd_table.append(dd_1)
    for k in range(2, 5): # build the k-th order difference table
        dd_k = np.zeros((n_total+4-k, k+1))
        dd_last = dd_table[-1]
        dd_k[:,0] = -dd_last[:-1,0]
        dd_k[:,-1] = dd_last[1:,-1]
        for j in range(1, k):
            dd_k[:,j] = dd_last[1:,j-1] - dd_last[:-1, j]
        dd_k /= (xi_c[k:] - xi_c[:-k])[:,np.newaxis]
        dd_table.append(dd_k)

    # 3. set initial values
    h = 1 - np.exp(4.0*(xi_c_fluid-1))
    h *= 1.0 + 0.2 * np.cos(20*xi_c_fluid)
    h[-1] = -h[-2]
    g = np.zeros((n_total+3, ))

    # 4. build the negative Laplacian for the liquid: (n_total+3, n_fluid+3)
    val_diag = np.zeros((n_fluid+3, ))
    val_up1 = np.zeros((n_fluid+2, ))
    val_lo1 = np.zeros((n_fluid+3, ))
    val_diag[2:-1] = -2.0 * dd_table[2][1:1+n_fluid, 1]
    val_up1[2:] = -2.0 * dd_table[2][1:1+n_fluid, 2]
    val_lo1[1:-2] = -2.0 * dd_table[2][1:1+n_fluid, 0]
    L4h = sp.diags((val_diag, val_up1, val_lo1), (0, 1, -2), (n_total+3, n_fluid+3), "csr")

    # 7. build the ghost matrix for the liquid
    val_lo1, val_up1 = val_up1, val_lo1; val_lo1[:] = 0.0; val_up1[:] = 0.0; val_diag[:] = 0.0
    val_diag[-1] = 0.5; val_lo1[-1] = 0.5
    G4hg = sp.diags((val_diag, val_lo1), (0, -1), (n_fluid+3, n_total+3), "csr")

    val_lo1 = np.zeros((n_fluid+2, ))
    val_up2 = np.zeros((n_fluid+1, ))
    val_diag[:] = 0.0; val_up1[:] = 0.0
    val_up1[0] = -1.0; val_up2[0] = 1.0 # symmetry at x=0
    val_lo1[0] = -1.0; val_up2[1] = 1.0 # symmetry at x=0
    val_lo1[-1] = 0.5; val_diag[-1] = 0.5 # Dirichlet at the CL
    G4hh = sp.diags((val_diag, val_up1, val_up2, val_lo1), (0, 1, 2, -1), (n_fluid+3, n_fluid+3), "csr")

    # 8. Some identities
    val_diag[:] = 0.0; val_diag[2:2+n_fluid] = 1.0
    Ihh = sp.diags((val_diag, ), (0, ), (n_fluid+3, n_fluid+3), "csr") # (n_fluid+3, n_fluid+3)
    Ihg = sp.diags((val_diag, ), (0, ), (n_fluid+3, n_total+3), "csr") # (n_fluid+3, n_total+3)
    
    # 4. build the negative Laplacian operator for the sheet
    # the rightmost cell is simply not needed; the last interior cell and the ghost next to it are set to zero. 
    val_up1 = np.zeros((n_total+2, ))
    val_lo1 = np.zeros((n_total+2, ))
    val_diag = np.zeros((n_total+3, ))
    val_up1[2:-1] = -2 * dd_table[2][1:-2,2]
    val_lo1[1:-2] = -2 * dd_table[2][1:-2,0]
    val_diag[2:-2] = -2 * dd_table[2][1:-2,1]
    pc_gamma = np.zeros((n_total+3, ))
    pc_gamma[2:2+n_fluid] = phyp.gamma[0]           # wet surface tension
    pc_gamma[2+n_fluid:2+n_total] = phyp.gamma[1] # dry surface tension
    gammaL = sp.diags((val_diag * pc_gamma, val_up1 * pc_gamma[:-1], val_lo1 * pc_gamma[1:]), (0, 1, -1), (n_total+3, n_total+3), "csr")

    # 5. build the biharmonic operator for the sheet
    val_up2 = np.zeros((n_total+1, ))
    val_lo2 = np.zeros((n_total+1, ))
    val_up1[:] = 0.0; val_lo1[:] = 0.0; val_diag[:] = 0.0
    val_up1[2:-1] = 24.0 * dd_table[4][:-1,3]
    val_up2[2:] = 24.0 * dd_table[4][:-1,4]
    val_lo1[1:-2] = 24.0 * dd_table[4][:-1,1]
    val_lo2[:-2] = 24.0 * dd_table[4][:-1,0]
    val_diag[2:-2] = 24.0 * dd_table[4][:-1,2]
    LL = sp.diags((val_diag, val_up1, val_up2, val_lo1, val_lo2), (0, 1, 2, -1, -2), (n_total+3, n_total+3), "csr")

    # 6. build the ghost matrix for the sheet
    val_up1[:] = 0.0; val_up2[:] = 0.0; val_diag[:] = 0.0; val_lo1[:] = 0.0; val_lo2[:] = 0.0
    val_up1[0] = -1.0; val_up2[0] = 1.0 # symmetry at x=0
    val_lo1[0] = -1.0; val_up2[1] = 1.0 # symmetry at x=0
    val_diag[-2] = 1.0 # symmetry at x=2 and fixed displacement
    val_diag[-1] = 1.0 # symmetry at x=2 and fixed displacement
    G = sp.diags((val_diag, val_up1, val_up2, val_lo1, val_lo2), (0, 1, 2, -1, -2), (n_total+3, n_total+3), "csr")
    del val_up1, val_up2, val_lo1, val_lo2, val_diag

    pyplot.ion()
    ax = pyplot.subplot()

    a = 1.0
    numSteps = ceil(solp.Te / solp.dt)
    for m in range(numSteps):
        # 1. assemble the fourth-order thin film operator for h
        # interpolate to cell boundaries
        h_mid = (h[2:-2] + h[3:-1]) / 2                  # (n_fluid-1, )
        g_mid = (g[2:2+n_fluid-1] + g[3:2+n_fluid]) / 2  # (n_fluid-1, )
        # calculate the flux coefficients at the cell boundaries
        fc = h_mid*(h_mid**2-g_mid**2) / 2 - (h_mid**3 - g_mid**3) / 6  - (g_mid/2 + phyp.slip) * (h_mid - g_mid)**2
        fc = np.concatenate(((0.0,), fc, (0.0, )))  # (n_fluid+1, ), zero flux at both boundaries
        # build the FD scheme
        ctab = np.zeros((n_fluid, 5))
        ctab[:,0] = -fc[:-1] * dd_table[3][:n_fluid, 0]   # (n_fluid, )
        ctab[:,4] = fc[1:] * dd_table[3][1:n_fluid+1, 3]  # (n_fluid, )
        for j in range(1, 4):
            ctab[:,j] = fc[1:] * dd_table[3][1:n_fluid+1,j-1] - fc[:-1] * dd_table[3][:n_fluid,j] # (n, )
        ctab = 6 * ctab / (xi_b_f[1:] - xi_b_f[:-1])[:,np.newaxis]
        # assemble the sparse matrix; dispose the rightmost ghost cell
        v_diag = np.zeros((n_fluid+3, ))
        v_diag[2:-1] = ctab[:,2]
        v_up1 = np.zeros((n_fluid+2, ))
        v_up1[2:] = ctab[:,3]
        v_lo1 = np.zeros((n_fluid+2, ))
        v_lo1[1:-1] = ctab[:,1]
        v_up2 = np.zeros((n_fluid+1, ))
        v_up2[2:] = ctab[:-1,4]
        v_lo2 = np.zeros((n_fluid+1, ))
        v_lo2[:-1] = ctab[:,0]
        C = sp.diags((v_diag, v_up1, v_up2, v_lo1, v_lo2), (0, 1, 2, -1, -2), (n_fluid+3, n_fluid+3), "csr")

        # calculate the CL speed
        tan_alpha = (h[-2] - h[-1]) / (a * dxi_at_cl)
        assert tan_alpha >= 0.
        tan_beta = (g[n_fluid+2] - g[n_fluid]) / (a * dxi_at_cl)
        assert tan_beta >= 0.
        theta_d = np.arctan(tan_alpha) + np.arctan(tan_beta)
        adot = phyp.mu_cl * 0.5 * (theta_d**2 - phyp.theta_Y**2)

        # calculate the advection term using upwind
        adv = np.zeros((n_fluid+3, ))
        if adot >= 0.0:
            adv[2:-1] = ((h[3:] - h[2:-1]) - (g[3:3+n_fluid] - g[2:2+n_fluid])) / (xi_c_fluid[3:] - xi_c_fluid[2:-1]) # (n_fluid, )
        else:
            adv[2:-1] = ((h[2:-1] - h[1:-2]) - (g[2:2+n_fluid] - g[1:1+n_fluid])) / (xi_c_fluid[2:-1] - xi_c_fluid[1:-2]) # (n_fluid, )
        adv[2:-1] *= xi_c_fluid[2:-1] * adot / a

        # assemble the linear system
        # A = sp.bmat((
        #     (Ihh + (solp.dt*phyp.gamma[2]/a**4)*C + G4hh, -Ihg-G4hg), 
        #     (phyp.gamma[2]*L4h/a**2, phyp.bm*LL/a**4+gammaL/a**2+G)
        #     ), format="csr")
        # # incorporate the jump
        # dummy_jump = np.zeros((n_total+3, ))
        # dummy_jump[2:] = np.minimum(-phyp.gamma[2] * theta_d / 6 * ((xi_c[2:-1] - 1.0) * a)**3, 0.0)
        # b = np.concatenate((solp.dt * adv, LL @ dummy_jump / a**4))
        # x = spsolve(A, b)
        # h_next = x[:n_fluid+3]
        # g_next = x[n_fluid+3:]
        h_g = np.zeros_like(h)
        h_g[2:-1] = h[2:-1] - g[2:n_fluid+2]
        h_next = spsolve(Ihh + (solp.dt * phyp.gamma[2]/a**4)*C + G4hh, solp.dt * adv + h_g)
        g_next = np.zeros_like(g)
        a += adot * solp.dt

        # some info
        assert np.all(h_next[2:-1] >= g_next[2:2+n_fluid])
        vol = np.sum((h_next[2:-1] - g_next[2:2+n_fluid]) * a * (xi_b_f[1:] - xi_b_f[:-1]))
        delta_h = np.linalg.norm(h[2:-1] - h_next[2:-1], ord=np.inf) / solp.dt
        delta_g = np.linalg.norm(g[2:-1] - g_next[2:-1], ord=np.inf) / solp.dt
        print("t = {:.6f}, step = {}/{}, diff = {:.2e}, {:.2e}, vol = {:.4e}, a = {:.5f}, adot = {:.2e}".format(
            (m+1)*solp.dt, m, numSteps, delta_h, delta_g, vol, a, adot))

        h[:] = h_next
        g[:] = g_next

        #
        ax.clear()
        ax.plot(a * xi_c_fluid[2:], h[2:], '-')
        ax.plot(a * xi_c[2:-2], g[2:-1], '--')
        ax.set_xlim(0.0, 2.0); ax.set_ylim(-0.1, 1.9); # ax.axis("equal")
        pyplot.draw(); pyplot.pause(1e-4)

    pyplot.ioff()
    pyplot.show()