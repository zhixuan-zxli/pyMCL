import numpy as np
from scipy import sparse as sp
from scikits.umfpack import spsolve
from matplotlib import pyplot
from runner import *

# test Poisson equation
# def testFiniteDifference(xi_b: np.ndarray):
#     for m in range(maxStep):
#         # a = 1.0 + m * dt * adot
#         # fill the ghosts near the symmetric boundary at x=0 and at x=a
#         h[0], h[1] = h[3], h[2]
#         h[-2] = -h[-3]
#         # interpolate to cell boundaries
#         h_mid = (h[3:-2] + h[2:-3]) / 2 # (n-1, )
#         # calculate the flux coefficients at the cell boundaries
#         fc = np.concatenate(((0.0,), h_mid**2 * (h_mid/3 + slip), (0.0,))) # (n+1, ), zero flux at both boundaries
#         # build the FD scheme
#         fdtab = np.zeros((n, 5))
#         fdtab[:,0] = - fc[:-1] * dd_table[3][:-1, 0] # (n, )
#         fdtab[:,4] = fc[1:] * dd_table[3][1:, 3]     # (n, )
#         for j in range(1, 4):
#             fdtab[:,j] = fc[1:] * dd_table[3][1:,j-1] - fc[:-1] * dd_table[3][:-1,j] # (n, )
#         fdtab = 6 * fdtab / (xi_b[1:] - xi_b[:-1])[:,np.newaxis]
#         # assemble the sparse matrix; dispose the rightmost ghost cell
#         v_diag = np.zeros((n+3, ))
#         v_diag[2:-1] = fdtab[:,2]
#         v_up1 = np.zeros((n+2, ))
#         v_up1[2:] = fdtab[:,3]
#         v_lo1 = np.zeros((n+2, ))
#         v_lo1[1:-1] = fdtab[:,1]
#         v_up2 = np.zeros((n+1, ))
#         v_up2[2:] = fdtab[:-1,4]
#         v_lo2 = np.zeros((n+1, ))
#         v_lo2[:-1] = fdtab[:,0]
#         LL = sp.diags((v_diag, v_up1, v_up2, v_lo1, v_lo2), (0, 1, 2, -1, -2), (n+3, n+3), "csr")
#         # assemble the boundary conditions
#         v_diag[:] = 0.0; v_up1[:] = 0.0; v_up2[:] = 0.0; v_lo1[:] = 0.0; v_lo2[:] = 0.0
#         v_up1[0] = -1.0; v_up2[0] = 1.0 # symmetry
#         v_lo1[0] = -1.0; v_up2[1] = 1.0 # symmetry
#         v_lo1[-1] = 0.5; v_diag[-1] = 0.5 # zero boundary condition
#         G = sp.diags((v_diag, v_up1, v_up2, v_lo1, v_lo2), (0, 1, 2, -1, -2), (n+3, n+3), "csr")
#         v_diag[:] = 0.0; v_diag[2:-1] = 1.0
#         I = sp.diags((v_diag, ), (0, ), (n+3, n+3), "csr")
#         del v_diag, v_up1, v_up2, v_lo1, v_lo2
#         # calculate the CL speed
#         tan_theta_d = -2*h[-3] / (a * dxi[-1])
#         theta_d = np.arctan(-tan_theta_d)
#         adot = beta * 0.5 * (theta_d**2 - theta_Y**2)
#         # calculate the advection term using upwind
#         adv = np.zeros((n, ))
#         if adot >= 0.0:
#             adv[:] = (h[3:-1] - h[2:-2]) / (xi_c[3:-1] - xi_c[2:-2]) # (n, )
#         else:
#             adv[:] = (h[2:-2] - h[1:-3]) / (xi_c[2:-2] - xi_c[1:-3]) # (n, )
#         adv *= xi_c[2:-2] * adot / a
#         # assemble the linear system
#         A = I + (dt / (Ca * a**4)) * LL + G
#         b = np.zeros((n+3, ))
#         b[2:-1] = h[2:-2] + dt * adv
#         h_next = spsolve(A, b)        
#         delta_h = np.linalg.norm(h_next[2:-1] - h[2:-2], np.inf) / dt
#         h[:-1] = h_next
#         a += adot * dt
#         # some info
#         vol = np.sum(h_next[2:-1] * a * (xi_b[1:] - xi_b[:-1]))
#         assert np.all(h[2:-2] >= 0.)
#         #
#         ax.clear()
#         ax.plot(a * xi_c[2:-2], h[2:-2], '-')
#         ax.plot((0.0, 2.0), (0.0, 0.0), '-')
#         print("t = {:.6f}, step = {}/{}, diff = {:.2e}, vol = {:.4e}, a = {:.5f}, adot = {:.2e}".format((m+1)*dt, m, maxStep, delta_h, vol, a, adot)); 
#         ax.set_xlim(0.0, 1.5); ax.set_ylim(-0.1, 1.4); # ax.axis("equal")
#         pyplot.draw(); pyplot.pause(1e-4)


#     pyplot.ioff()
#     print("Finished.")

#         del val_up1, val_up2, val_up3, val_lo1, val_lo2, val_lo3, val_diag

#         # account for the jump
#         c = np.where(xi_c >= 0.5, 0.5 * 1.0/6*(xi_c - 0.5)**3, 0.0)
#         Ac = A @ c

@dataclass
class PhysicalParameters:
    gamma: tuple[float] = (5.0, 5.0, 1.0) # the (effective) surface tension for the wet, dry and the interface
    # Ca: float = gamma[-1]
    slip: float = 1e-4   # the slip length
    theta_Y: float = 1.0
    mu_cl: float = 1.0
    bm: float = 1e-4     # the bending modulus

solp = SolverParameters(dt = 1/1024**2, Te=1.0)

if __name__ == "__main__":
    
    # 1. set up the grid. 
    m = 64
    xi_b_f = np.concatenate((
        np.linspace(0.0, 0.5, m+1), 
        np.linspace(1/2, 3/4, m+1)[1:], 
        np.linspace(3/4, 7/8, m+1)[1:],
        np.linspace(7/8, 15/16, m+1)[1:],
        np.linspace(15/16, 31/32, m+1)[1:],
        np.linspace(31/32, 63/64, m+1)[1:],
        np.linspace(63/64, 1.0, 2*m+1)[1:],
    ))
    xi_b = np.concatenate((xi_b_f, 2.0 - xi_b_f[1:])) # cell boundaries
    n_fluid = xi_b_f.size - 1 # number of cells for the fluid (excluding ghosts)
    n_total = xi_b.size - 1   # number of cells total (excluding ghosts)
    
    dxi = xi_b[1:] - xi_b[:-1] # mesh step size on the reference domain
    xi_g = np.concatenate(((xi_b[0] - 2*dxi[0], xi_b[0] - dxi[0]), xi_b, (xi_b[-1] + dxi[-1], xi_b[-1] + 2*dxi[-1]))) # add ghost points
    xi_c = (xi_g[:-1] + xi_g[1:]) / 2 # get the cell centers (with ghosts)
    xi_c_fluid = xi_c[:n_fluid+3]     # the fluid cells, with ghosts
    
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
    h = 1 - np.exp(10.0*(xi_c_fluid-1))
    h *= 1.0 + 0.05 * np.cos(20*xi_c_fluid)
    g = np.zeros_like(xi_c)
    
    # 4. build the negative Laplacian operator for the sheet
    # the rightmost cell is simply not needed; the last interior cell and the ghost next to it are set to zero. 
    val_up2 = np.zeros((n_total+1, ))
    val_lo2 = np.zeros((n_total+1, ))
    val_up1 = np.zeros((n_total+2, ))
    val_lo1 = np.zeros((n_total+2, ))
    val_diag = np.zeros((n_total+3, ))
    val_up1[2:-1] = -2 * dd_table[2][1:-2,2]
    val_lo1[1:-2] = -2 * dd_table[2][1:-2,0]
    val_diag[2:-2] = -2 * dd_table[2][1:-2,1]
    pc_gamma = np.zeros((n_total+3, ))
    pc_gamma[2:2+n_fluid] = solp.gamma[0]           # wet surface tension
    pc_gamma[2+n_fluid:2+n_total-1] = solp.gamma[1] # dry surface tension
    gammaL = sp.diags((val_diag * pc_gamma, val_up1 * pc_gamma[:-1], val_lo1 * pc_gamma[1:]), (0, 1, -1), (n_total+3, n_total+3), "csr")

    # 5. build the biharmonic operator for the sheet
    val_up1[2:-1] = 24.0 * dd_table[4][:-1,3]
    val_up2[2:] = 24.0 * dd_table[4][:-1,4]
    val_lo1[1:-2] = 24.0 * dd_table[4][:-1,1]
    val_lo2[:-2] = 24.0 * dd_table[4][:-1,0]
    val_diag[2:-2] = 24.0 * dd_table[4][:-1,2]
    LL = sp.diags((val_diag, val_up1, val_up2, val_lo1, val_lo2), (0, 1, 2, -1, -2), (n_total+3, n_total+3), "csr")

    # 6. build the ghost matrix for the sheet
    val_up1[:] = 0.0; val_up2[:] = 0.0; val_diag[:] = 0.0; val_lo1[:] = 0.0; val_lo2[:] = 0.0
    val_up1[0] = -1.0; val_up2[0] = 1.0 # symmetry at x=0
    val_lo1[0] = -1.0; val_up2[2] = 1.0 # symmetry at x=0
    val_diag[-2] = 1.0 # symmetry at x=2 and fixed displacement
    val_diag[-1] = 1.0 # symmetry at x=2 and fixed displacement
    G = sp.diags((val_diag, val_up1, val_up2, val_lo1, val_lo2), (0, 1, 2, -1, -2), (n_total+3, n_total+3), "csr")
    del val_up1, val_up2, val_lo1, val_lo2, val_diag

