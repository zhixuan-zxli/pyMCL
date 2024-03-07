import numpy as np
from mesh import Mesh
from fe import Measure, TriP2
from function import Function
from assemble import assembler, Form, setMeshMapping
# from matplotlib import pyplot


def functional(coord) -> np.ndarray:
    x, y = coord[0], coord[1] # (Ne, Nq)
    f = np.sin(np.pi*x) * np.cos(y)
    return f[np.newaxis] * coord.dx
    

def linear(psi, coord) -> np.ndarray:
    # psi: (1,1,Nq)
    # coord: (2,Ne,Nq)
    x, y = coord[0], coord[1]
    data = np.sin(np.pi*x) * np.cos(y) # (Ne, Nq)
    data = data[np.newaxis] * psi * coord.dx
    return data

if __name__ == "__main__":
    mesh = Mesh()
    mesh.load("mesh/unit_square.msh")
    setMeshMapping(mesh)
    P2 = TriP2(mesh)
    asm = assembler(P2, None, Measure(2, None), 4)
    a = asm.functional(Form(functional, "f"))
    v = asm.linear(Form(linear, "f"))
    print(a, v.sum())
    # pyplot.figure()
    # mesh.draw()
    # pyplot.show()
    pass
