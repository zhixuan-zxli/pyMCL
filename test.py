import numpy as np
import fem
from matplotlib import pyplot

if __name__ == "__main__":
    mesh = fem.Mesh()
    mesh.load("mesh/unit_square.msh")
    u_space = fem.FiniteElement(mesh, fem.element.LagrangeTri(2))
    # u_space = fem.FESpace(mesh, fem.Element("Lagrange", "tri", 2))
    # qpts = np.array([[1.0/2, 0.0, 0.0], [1.0/2, 1.0/2, 0.0], [0.0, 1.0/2, 0.0]])
    # qpts[:,-1] = 1.0 - np.sum(qpts, axis=1)
    # qpts = qpts.T
    # phi, dphidx, dphidy = u_space.elem.basis(qpts)
    # import element
    # elem = element.LagrangeTri(2)
    # elem.eval_basis(5, np.array([1]))
    # print(elem)
    pass
