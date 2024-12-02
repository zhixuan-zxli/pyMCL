import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve

class BendingProblem:
    """
    Solve the one-dimensional bending problem with a jump condition at the origin.
    B*y'''' - \gamma*y'' = f(x), 
    B [y'''] = J at x=0. 
    """
    bm: float
    gamma: tuple[float]
    domain: tuple[float]
    num_cell: int

    x: np.ndarray # the cell center coordinates (without ghosts)

    def __init__(self, bm: float, gamma: tuple[float], domain: tuple[float], num_cell: int) -> None:
        self.bm = bm
        self.gamma = gamma
        self.domain = domain
        self.num_cell = num_cell
        self.dx = (domain[1] - domain[0]) / num_cell
        self.x = np.linspace(domain[0] + 0.5*self.dx, domain[1] - 0.5*self.dx, num_cell)
        self.i_idx = np.searchsorted(self.x, 0.0)
        assert 1 <= self.i_idx <= num_cell

    def setupLinearSystem(self) -> None:
        dx = self.dx
        n = self.num_cell
        # build the negative Laplacian, with Dirichlet condition at the left boundary
        val_dia = np.zeros((n+2, ))
        val_up1 = np.zeros((n+1, ))
        val_lo1 = np.zeros((n+1, ))
        val_dia[1:-1] = 2.0 / dx**2
        val_up1[1:] = -1.0 / dx**2
        val_lo1[:-1] = -1.0 / dx**2
        val_dia[0] = 1.0
        self.L = sp.diags([val_lo1, val_dia, val_up1], [-1, 0, 1], shape=(n+2, n+2), format="csc")
        # build the diagonal of gamma
        val_dia[:] = 0.0
        val_dia[1:self.i_idx+1] = self.gamma[0]
        val_dia[self.i_idx+1:-1] = self.gamma[1]
        self.dia = sp.diags([val_dia], [0], shape=(n+2, n+2), format="csc")
        # build the diagonal of identity
        val_dia[:] = 0.0
        val_dia[1:-1] = 1.0
        self.I = sp.diags([val_dia], [0], shape=(n+2, n+2), format="csc")
        # build the ghost matrix for Neumann-type condition at the right boundary
        val_dia[:] = 0.0; val_lo1[:] = 0.0
        val_lo1[-1] = -1.0 / dx
        val_dia[-1] = 1.0 / dx
        self.GN = sp.diags([val_dia, val_lo1], [0, -1], shape=(n+2, n+2), format="csc")
        # build the ghost matrix for Dirichlet-type condition at the right boundary
        val_dia[-1] = 1.0
        self.GD = sp.diags([val_dia], [0], shape=(n+2, n+2), format="csc")

        # build the correction for the jump condition
        jv = np.zeros((n+2, ))
        jv[1:-1] = np.maximum(self.x, 0.0)
        self.corr = self.L @ jv # the effective entries are i_idx and i_idx+1
        self.corr[:self.i_idx] = 0.0; self.corr[self.i_idx+2:] = 0.0
        # col_vec = col_vec[self.i_idx:self.i_idx+2]
        # row_vec = (-1.0 / dx, 1.0 / dx)
        # # build the outer product with rows (i_idx, i_idx+1) and columns (i_idx, i_idx+1)
        # ind = np.array((self.i_idx, self.i_idx+1), dtype=np.int)
        # row_indices = np.tile(ind, 2)
        # col_indices = np.repeat(ind, 2)
        # vals = (col_vec[np.newaxis] * row_vec[:, np.newaxis]).flatten()
        # self.J4ky = sp.csc_matrix((vals, (row_indices, col_indices)), shape=(n+2, n+2))

    def solve(self, f: np.ndarray, c: tuple[float]) -> tuple[np.ndarray]:
        """
        c: [0] for the Neumann data at the right, [1] for the jump condition at the origin, multiplied by bm.
        """
        n = self.num_cell
        # assemble the matrix
        A = sp.bmat(((self.L + self.GN, self.I), 
                     (None, self.bm * self.L + self.dia + self.GD)), 
                     format="csc")
        # assemble the right-hand-side
        b = np.zeros((2*n+4, ))
        b[n+1] = c[0]
        b[n+3:2*n+3] = -f + c[1] * self.corr[1:-1]
        # solve the linear system
        sol = spsolve(A, b)
        y = sol[1:n+1]
        k = sol[n+3:2*n+3]
        return (y, k)
