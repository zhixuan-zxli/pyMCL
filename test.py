from fem.mesh import Mesh

if __name__ == "__main__":
    mesh = Mesh()
    mesh.load("mesh/unit_square.msh")
    mesh.build_topology()
    pass
