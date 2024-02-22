import meshio
import numpy as np

class Mesh:
    def __init__(self, mesh_name: str) -> None:
        msh = meshio.read(mesh_name)
        self.Np = [msh.points.shape[0]]
        # 1. read the points
        self.point = msh.points
        if np.all(self.point[:,-1] == 0.): 
            self.point = self.point[:,:-1] # Remove the z coordinates if this is a planar mesh
        self.point = np.hstack((self.point, np.zeros((self.Np[0], 1)))) # append the tags

        # 2. read the higher-dimensional entities
        num_cells = len(msh.cells)
        has_physical = "gmsh:physical" in msh.cell_data
        self.edge = np.zeros((0, 3), dtype=np.uint32)
        self.tri = np.zeros((0, 4), dtype=np.uint32)

        for cid in range(num_cells):
            cell = msh.cells[cid]
            if has_physical:
                data = msh.cell_data["gmsh:physical"][cid].reshape(-1, 1) # reshape to a column vector
            else:
                data = np.zeros((cell.data.shape[0], 1), dtype=np.uint32)
            if cell.type == "vertex" and data != None:
                self.point[cell.data, :] = data
            elif cell.type == "line":
                self.edge = np.vstack((self.edge, np.hstack((cell.data, data))))
            elif cell.type == "triangle":
                self.tri = np.vstack((self.tri, np.hstack((cell.data, data))))
            else:
                raise RuntimeError("Unrecognized cell type. ")

