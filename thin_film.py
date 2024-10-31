from typing import Union
import numpy as np
from scipy import sparse as sp
from scikits.umfpack import spsolve
from matplotlib import pyplot
from runner import *

@dataclass
class PhysicalParameters:
    gamma: tuple[float] = (4.0, 4.9, 1.0) # the (effective) surface tension for the wet, dry and the interface
    slip: float = 1e-2   # the slip length
    theta_Y: float = 1.0
    mu_cl: float = 1.0
    bm: float = 1 * 1e-2     # the bending modulus

class ThinFilmRunner(Runner):

    def __init__(self, solp):
        super().__init__(solp)

    def prepare(self, base_grid: Union[int, np.ndarray]) -> None:
        """
        base_grid [int]  number of cells for the fluid, before refinement; 
          or the cell boundaries in [0,1]. 
        """
        super().prepare()

        self.phyp = PhysicalParameters()
        with open(self._get_output_name("PhysicalParameters"), "wb") as f:
            pickle.dump(self.phyp, f)
            
        if isinstance(base_grid, int):
            xi_b_f = np.linspace(0.0, 1.0, base_grid * 2**solp.spaceref + 1)
        else:
            xi_b_f = base_grid
        
        xi_b = np.concatenate((xi_b_f, 2.0 - xi_b_f[-2::-1])) # cell boundaries
        n_fluid = xi_b_f.size - 1 # number of cells for the fluid (excluding ghosts)
        n_total = xi_b.size - 1   # number of cells total (excluding ghosts)
        self.xi_b, self.xi_b_f = xi_b, xi_b_f
        dxi = xi_b[1:] - xi_b[:-1] # mesh step size on the reference domain
        xi_g = np.concatenate(((xi_b[0] - 2*dxi[0], xi_b[0] - dxi[0]), xi_b, (xi_b[-1] + dxi[-1], xi_b[-1] + 2*dxi[-1]))) # add ghost points
        xi_c = (xi_g[:-1] + xi_g[1:]) / 2 # get the cell centers (with 2 ghosts)
        xi_c_f = xi_c[:n_fluid+3]         # the fluid cells, with one ghost on the right end
        self.min_dxi = np.min(dxi)
        self.dxi_at_cl = xi_b_f[-1] - xi_b_f[-2] #xi_c[n_fluid+2] - xi_c[n_fluid+1]
        self.xi_c, self.xi_c_f = xi_c, xi_c_f
        
        # build the divided difference table. 
        dd_table = [None]
        dd_1 = np.zeros((n_total+3, 2)) # first-order difference
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
        self.dd_table = dd_table

        # build the negative Laplacian for the fluid: (n_total+2, n_fluid+3)
        val_dia = np.zeros((n_fluid+3, ))
        val_up1 = np.zeros((n_fluid+2, ))
        val_up2 = np.zeros((n_fluid+1, ))
        # val_lo1 = np.zeros((n_fluid+3, ))
        val_dia[1:1+n_fluid] = -2.0 * dd_table[2][1:1+n_fluid, 0]
        val_up1[1:1+n_fluid] = -2.0 * dd_table[2][1:1+n_fluid, 1]
        val_up2[1:1+n_fluid] = -2.0 * dd_table[2][1:1+n_fluid, 2]
        self.L4h = sp.diags((val_dia, val_up1, val_up2), (0, 1, 2), (n_total+2, n_fluid+3), "csc")

        # build the ghost matrix for the fluid-sheet: (n_fluid+3, n_total+2)
        val_lo1 = np.zeros((n_fluid+2, ))
        val_lo2 = np.zeros((n_fluid+1, ))
        val_lo2[-1] = 0.5; val_lo1[-1] = 0.5
        self.G4hg = sp.diags((val_lo2, val_lo1), (-2, -1), (n_fluid+3, n_total+2), "csc")

        # ghost matrix for fluid-fluid: (n_fluid+3, n_fluid+3)
        val_dia[:] = 0.0; val_up1[:] = 0.0; val_up2[:] = 0.0; val_lo1[:] = 0.0
        val_up1[0] = -1.0; val_up2[0] = 1.0 # symmetry at x=0
        val_lo1[0] = -1.0; val_up2[1] = 1.0 # symmetry at x=0
        val_lo1[-1] = 0.5; val_dia[-1] = 0.5 # Dirichlet at the CL
        self.G4hh = sp.diags((val_dia, val_up1, val_up2, val_lo1), (0, 1, 2, -1), (n_fluid+3, n_fluid+3), "csc")

        # Some identity matrices for the time stepping
        val_dia[:2] = 0.0; val_dia[2:2+n_fluid] = 1.0; val_dia[2+n_fluid:] = 0.0
        self.Ihh = sp.diags((val_dia, ), (0, ), (n_fluid+3, n_fluid+3), "csc") # (n_fluid+3, n_fluid+3)
        val_lo1[0] = 0.0; val_lo1[1:1+n_fluid] = 1.0; val_lo1[1+n_fluid:] = 0.0
        self.Ihg = sp.diags((val_lo1, ), (-1, ), (n_fluid+3, n_total+2), "csc") # (n_fluid+3, n_total+2)
        
        # build the negative Laplacian operator for the sheet: (n_total+2, n_total+2)
        val_up1 = np.zeros((n_total+1, ))
        val_lo1 = np.zeros((n_total+1, ))
        val_dia = np.zeros((n_total+2, ))
        val_lo1[:-1] = -2 * dd_table[2][1:-1,0]
        val_dia[1:-1] = -2 * dd_table[2][1:-1,1]
        val_up1[1:] = -2 * dd_table[2][1:-1,2]
        self.L = sp.diags((val_dia, val_up1, val_lo1), (0, 1, -1), (n_total+2, n_total+2), "csc")

        # build the piecewise constant gamma: (n_total+2, n_total+2)
        val_dia[0] = 0.0
        val_dia[1:1+n_fluid] = self.phyp.gamma[0] # wet surface tension
        val_dia[1+n_fluid:-1] = self.phyp.gamma[1] # dry surface tension
        val_dia[-1] = 0.0
        self.gamma_dia = sp.diags((val_dia, ), (0, ), (n_total+2, n_total+2), "csc")

        val_dia[1:-1] = 1.0
        self.Igk = sp.diags((val_dia, ), (0, ), (n_total+2, n_total+2), "csc")

        # build the ghost matrix for the sheet: (n_total+2, n_total+2)
        val_up1[:] = 0.0; val_dia[:] = 0.0; val_lo1[:] = 0.0
        val_dia[0] = 1.0; val_up1[0] = -1.0  # symmetry at x=0
        val_dia[-1] = 1.0; val_lo1[-1] = 1.0 # Dirichlet at farfield
        self.G = sp.diags((val_dia, val_up1, val_lo1), (0, 1, -1), (n_total+2, n_total+2), "csc")
        del val_up1, val_up2, val_lo1, val_lo2, val_dia

        if self.args.vis:
            pyplot.ion()
            self.ax = pyplot.subplot()

        # 3. set initial values
        self.t = 0.0
        self.a = 1.0
        self.a_hist = np.zeros((2, self.num_steps + 1))
        self.cp = 0 # number of checkpoints reached
        
        self.h = 1 - np.exp(4.0*(xi_c_f-1))
        # self.h *= 2.0
        self.h[-1] = -self.h[-2]
        self.g = np.zeros((n_total+2, ))
        self.kappa = np.zeros((n_total+2, ))

    def pre_step(self) -> None:
        # some info
        xi_c, xi_b_f = self.xi_c, self.xi_b_f
        n_fluid = xi_b_f.size - 1
        fluid_h = self.h[2:-1] >= self.g[1:1+n_fluid]
        assert np.all(fluid_h >= 0.0)
        vol = np.sum(fluid_h * self.a * (xi_b_f[1:] - xi_b_f[:-1]))
        print("t = {:.6f}, dt = {:.3e}, vol = {:.3e}, ".format(self.t, self.solp.dt, vol), end="")
        # save intermediate result
        self.a_hist[0, self.step] = self.t; self.a_hist[1, self.step] = self.a
        if self.t >= self.cp * self.solp.dt_cp:
            filename = self._get_output_name("{:04}.npz".format(self.cp))
            np.savez(filename, h=self.h, g=self.g, kappa=self.kappa, a_hist=self.a_hist[:, :self.step+1])
            self.cp += 1
        # visualization
        if self.args.vis:
            self.ax.clear()
            self.ax.plot(self.a * self.xi_c_f[2:], self.h[2:], '-')   # fluid
            # todo: curvature of the fluid
            self.ax.plot(self.a * xi_c[2:-2], self.g[1:-1], '-')      # sheet
            self.ax.plot(self.a * xi_c[2:-2], self.kappa[1:-1], '--') # curvature of the sheet
            self.ax.set_xlim(0.0, 3.0); self.ax.set_ylim(-1.0, 2.0)
            pyplot.draw(); pyplot.pause(1e-4)
        
        return self.t >= self.solp.Te
    
    def main_step(self) -> None:
           
        h, g = self.h, self.g
        dd_table = self.dd_table
        xi_b_f, xi_c_f, xi_c = self.xi_b_f, self.xi_c_f, self.xi_c
        n_total = self.xi_b.size - 1
        n_fluid = xi_b_f.size - 1
        dxi_at_cl = self.dxi_at_cl

        # 1. assemble the fourth-order thin film operator for h
        # interpolate to cell boundaries
        eta = (xi_b_f[1:-1] - xi_c_f[2:-2]) / (xi_c_f[3:-1] - xi_c_f[2:-2])
        h_mid = (1-eta) * h[2:-2] + eta * h[3:-1]              # (n_fluid-1, )
        g_mid = (1-eta) * g[1:n_fluid] + eta * g[2:n_fluid+1]  # (n_fluid-1, )
        # calculate the flux coefficients at the cell boundaries
        fc = (h_mid - g_mid) * ((h_mid - g_mid)**2/12 + (h_mid**2 + g_mid**2)/4 + self.phyp.slip * (h_mid - g_mid))
        fc = np.concatenate(((0.0,), fc, (0.0, )))  # (n_fluid+1, ), zero flux at both boundaries
        # build the FD scheme
        ctab = np.zeros((n_fluid, 5))
        ctab[:,0] = -fc[:-1] * dd_table[3][:n_fluid, 0]   # (n_fluid, )
        ctab[:,4] = fc[1:] * dd_table[3][1:n_fluid+1, 3]  # (n_fluid, )
        for j in range(1, 4):
            ctab[:,j] = fc[1:] * dd_table[3][1:n_fluid+1,j-1] - fc[:-1] * dd_table[3][:n_fluid,j] # (n, )
        ctab = 6 * ctab / (xi_b_f[1:] - xi_b_f[:-1])[:,np.newaxis]
        # assemble the matrix for the thin film operator: (n_fluid+3, n_fluid+3)
        v_dia = np.zeros((n_fluid+3, ))
        v_dia[2:-1] = ctab[:,2]
        v_up1 = np.zeros((n_fluid+2, ))
        v_up1[2:] = ctab[:,3]
        v_lo1 = np.zeros((n_fluid+2, ))
        v_lo1[1:-1] = ctab[:,1]
        v_up2 = np.zeros((n_fluid+1, ))
        v_up2[2:] = ctab[:-1,4]
        v_lo2 = np.zeros((n_fluid+1, ))
        v_lo2[:-1] = ctab[:,0]
        C = sp.diags((v_dia, v_up1, v_up2, v_lo1, v_lo2), (0, 1, 2, -1, -2), (n_fluid+3, n_fluid+3), "csc")

        # calculate the CL speed
        dh = (h[-1] - h[-2]) / (self.a * dxi_at_cl)
        dg = (g[n_fluid+1] - g[n_fluid]) / (self.a * dxi_at_cl)
        adot = self.phyp.mu_cl/2 * ((dh-dg)**2 - self.phyp.theta_Y**2)
        a_next = self.a + adot * solp.dt

        # calculate the advection term using upwind
        adv = np.zeros((n_fluid+3, ))
        # if adot >= 0.0:
        #     adv[2:-1] = ((h[3:] - h[2:-1]) - (g[3:3+n_fluid] - g[2:2+n_fluid])) / (xi_c_f[3:] - xi_c_f[2:-1]) # (n_fluid, )
        # else:
        #     adv[2:-1] = ((h[2:-1] - h[1:-2]) - (g[2:2+n_fluid] - g[1:1+n_fluid])) / (xi_c_f[2:-1] - xi_c_f[1:-2]) # (n_fluid, )
        # center in space
        adv[2:-1] = ((h[3:] - h[1:-2]) - (g[2:2+n_fluid] - g[:n_fluid])) / (xi_c_f[3:] - xi_c_f[1:-2]) # (n_fluid, )
        adv[2:-1] *= xi_c_f[2:-1] * adot / a_next
        
        # incorporate the jump
        gamma = self.phyp.gamma
        jump = -(gamma[2] * dh + (gamma[0]-gamma[1]) * dg)
        jv = np.zeros((n_total+2, ))
        jv[1:-1] = np.maximum(xi_c[2:-2] - 1.0, 0.0) 
        Ljv = (self.L @ jv) / a_next
        Ljv[n_fluid+2:] = 0.0

        # assemble the linear system
        #    h   g    kappa
        A = sp.bmat((
            (self.Ihh + (solp.dt*gamma[2]/a_next**4)*C + self.G4hh, -self.Ihg - self.G4hg, None), # h
            (None, self.L/a_next**2 + self.G, self.Igk), # g
            (-gamma[2]*self.L4h/a_next**2, None, self.phyp.bm*self.L/a_next**2 + self.gamma_dia + self.G),  # kappa
            ), format="csc")
        # prepare the RHS
        h_g = np.zeros_like(h)
        h_g[2:-1] = h[2:-1] - g[1:n_fluid+1]
        g0 = np.zeros_like(g)
        b = np.concatenate((solp.dt * adv + h_g, g0, jump * Ljv))
        x = spsolve(A, b)
        h_next = x[:n_fluid+3]
        g_next = x[n_fluid+3:n_fluid+n_total+5]
        kappa_next = x[n_fluid+n_total+5:]

        # some other info: 
        delta_h = np.linalg.norm(h_next[2:-1] - h[2:-1], ord=np.inf) / solp.dt
        delta_g = np.linalg.norm(g_next[1:-1] - g[1:-1], ord=np.inf) / solp.dt
        print("diff = {:.2e}, {:.2e}, a_next = {:.5f}, adot = {:.2e}".format(delta_h, delta_g, a_next, adot))

        self.h[:] = h_next
        self.g[:] = g_next
        self.kappa[:] = kappa_next
        self.a = a_next

        # adaptively change the dt
        self.t += self.solp.dt
        if self.solp.adapt_t and self.step > 0 and self.step % 128 == 0 \
            and solp.dt < self.min_dxi / adot / 16 and solp.dt < self.min_dxi:
            solp.dt *= 2

def downsample(u: np.ndarray) -> np.ndarray:
    usize = u.size
    u_down = np.zeros(((usize-3)//2 + 3, ))
    u_down[2:-1] = (u[2:-2:2] + u[3:-1:2]) / 2
    # symmetry condition at the left
    u_down[0] = u_down[3]
    u_down[1] = u_down[2]
    return u_down

if __name__ == "__main__":
    
    # set up the grid. 
    # m = 32
    # xi_b_f = np.concatenate((
    #     np.linspace(0.0, 0.5, m+1), 
    #     np.linspace(1/2, 3/4, m+1)[1:], 
    #     np.linspace(3/4, 7/8, m+1)[1:],
    #     np.linspace(7/8, 15/16, m+1)[1:],
    #     np.linspace(15/16, 31/32, m+1)[1:],
    #     np.linspace(31/32, 63/64, m+1)[1:],
    #     np.linspace(63/64, 1.0, 2*m+1)[1:],
    # ))

    solp = SolverParameters(dt = 1/(1024*32), Te=1.0)
    solp.dt_cp = 1.0/32
    solp.adapt_t = False

    runner = ThinFilmRunner(solp)
    # runner.prepare(base_grid=xi_b_f)
    runner.prepare(base_grid=1024)
    # read from file the initial conditions
    # initial_data = np.load("result/tf-sample-1024.npz") 
    # h, g = initial_data["h"], initial_data["g"]
    # assert runner.h.size <= h.size
    # while runner.h.size < h.size:
    #     h = downsample(h)
    #     g = downsample(g)
    # # set the boundary conditions at the right end
    # g[-1] = 0.0; g[-2] = 0.0
    # n_fluid = runner.n_fluid
    # h[n_fluid+2] = g[n_fluid+1] + g[n_fluid+2] - h[n_fluid+1]
    # runner.h = h
    # runner.g = g
    # runner.a = initial_data["a"]
    #
    runner.run()
    runner.finish()

    if runner.args.vis:
        pyplot.ioff()
        pyplot.show()
