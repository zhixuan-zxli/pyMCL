import numpy as np

class Element:
    def __init__(self, type: str, cell: str, degree: int) -> None:
        self.type = type
        self.cell = cell
        self.degree = degree

    def __str__(self) -> str:
        return "Degree-{} {} element on {}".format(self.degree, self.type, self.cell)

# qpts : (nd+1) * num_qpts
# output : num_qpts or nd * num_qpts

class LagrangeTri(Element):

    tdim: int = 2

    def __init__(self, degree: int) -> None:
        super().__init__("Lagrange", "triangle", degree)
        self.num_basis = {1: 3, 2: 6}[degree]
        self.eval_basis = {1: self._eval_basis_1, 2: self._eval_basis_2}[degree]
        self.eval_grad = {1: self._eval_grad_1, 2: self._eval_grad_2}[degree]

    def _eval_basis_1(self, basis_id:int, qpts: np.ndarray) -> np.ndarray:
        assert(basis_id < self.num_basis)
        x = qpts[0, :]
        y = qpts[0, :]
        if basis_id == 0:
            return 1.0 - x - y
        if basis_id == 1:
            return x
        if basis_id == 2:
            return y
    
    def _eval_grad_1(self, basis_id:int, qpts: np.ndarray) -> np.ndarray:
        assert(basis_id < self.num_basis)
        x = qpts[0, :]
        y = qpts[0, :]
        if basis_id == 0:
            return np.vstack((-np.ones_like(x), -np.ones_like(y)))
        if basis_id == 1:
            return np.vstack((np.ones_like(x), np.zeros_like(y)))
        if basis_id == 2:
            return np.vstack((np.zeros_like(x), np.zeros_like(y)))
    
    def _eval_basis_2(self, basis_id:int, qpts: np.ndarray) -> np.ndarray:
        assert(basis_id < self.num_basis)
        x = qpts[0, :]
        y = qpts[1, :]
        if basis_id == 0:
            return 2.0*x**2 - 3.0*x + 1.0 + 2.0*y**2 - 3.0*y + 4.0*x*y
        if basis_id == 1:
            return 2.0*x*(x-1.0/2)
        if basis_id == 2:
            return 2.0*y*(y-1.0/2)
        if basis_id == 3:
            return -4.0*x*(x+y-1)
        if basis_id == 4:
            return 4.0*x*y
        if basis_id == 5:
            return -4.0*y*(x+y-1)
    
    def _eval_grad_2(self, basis_id:int, qpts: np.ndarray) -> np.ndarray:
        assert(basis_id < self.num_basis)
        x = qpts[0, :]
        y = qpts[1, :]
        if basis_id == 0:
            return np.vstack((4.0*x+4.0*y-3.0, 4.0*x+4.0*y-3.0))
        if basis_id == 1:
            return np.vstack((4.0*x-1.0, 0.0*y))
        if basis_id == 2:
            return np.vstack((0.0*x, 4.0*y-1.0))
        if basis_id == 3:
            return np.vstack((-8.0*x-4.0*y+4.0, -4.0*x))
        if basis_id == 4:
            return np.vstack((4.0*y, 4.0*x))
        if basis_id == 5:
            return np.vstack((-4.0*y, -4.0*x-8.0*y+4.0))
    