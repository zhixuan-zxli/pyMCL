import numpy as np
from mesh import Mesh
from fe import Measure, TriP2
from function import Function
from assemble import assembler, setMeshMapping
# from matplotlib import pyplot

class f1:
    hint: str = "f"
    def __call__(self, coord, w) -> np.ndarray:
        x, y = coord[0], coord[1] # (Ne, Nq)
        return np.sin(np.pi*x) * np.cos(y) * coord.dx

if __name__ == "__main__":
    mesh = Mesh()
    mesh.load("mesh/unit_square.msh")
    setMeshMapping(mesh)
    P2 = TriP2(mesh)
    asm = assembler(P2, None, Measure(2, None), 4)
    u = Function(P2)
    print(asm.functional(f1()))
    # pyplot.figure()
    # mesh.draw()
    # pyplot.show()
    pass
