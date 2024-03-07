import numpy as np
from mesh import Mesh
from fe import Measure, TriP2
from function import Function
from assemble import assembler, setMeshMapping
# from matplotlib import pyplot

class f1:
    hint: str = "f"
    def __call__(self, coord) -> np.ndarray:
        x, y = coord[0], coord[1] # (Ne, Nq)
        f = np.sin(np.pi*x) * np.cos(y)
        return f[np.newaxis] * coord.dx
    
class lin1:
    hint: str = "f"
    def __call__(self, psi, coord) -> np.ndarray:
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
    a = asm.functional(f1())
    v = asm.linear(lin1())
    print(a, v.sum())
    # pyplot.figure()
    # mesh.draw()
    # pyplot.show()
    pass
