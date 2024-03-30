from fem.mesh import Mesh
from matplotlib import pyplot

if __name__ == "__main__":
    mesh = Mesh()
    mesh.load("mesh/two-phase.msh")

    b_mesh = mesh.view(1, (3, ))
    b_mesh.draw()
    pyplot.show()
    # mesh.draw()
    # pyplot.show()
    pass
