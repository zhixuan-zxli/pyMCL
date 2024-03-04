# import numpy as np
from mesh import Mesh
from fe import Measure, TriP2
from function import Function
from assemble import assembler, setMeshMapping
from matplotlib import pyplot

if __name__ == "__main__":
    mesh = Mesh()
    mesh.load("mesh/unit_square.msh")
    setMeshMapping(mesh)
    space = TriP2(mesh)
    dx = Measure(2, None)
    asm = assembler(space, None, dx, 4)
    pyplot.figure()
    mesh.draw()
    pyplot.show()
    pass
