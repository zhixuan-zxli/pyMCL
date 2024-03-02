import meshio
import numpy as np
from matplotlib import pyplot

class Mesh:
    def __init__(self) -> None:
        self.point = None
        self.point_tag = None
        self.cell = [
            None, 
            np.zeros((0, 3), dtype=np.uint32), # edge
            np.zeros((0, 4), dtype=np.uint32), # triangle
            np.zeros((0, 5), dtype=np.uint32)                    
        ]
        self.entities = [None] * 4
        self.gdim = 0
        self.tdim = 0

    def load(self, mesh_name: str) -> None:
        """
        Load a GMSH mesh. 
        """
        msh = meshio.read(mesh_name)
        # 1. read the points
        self.point = msh.points
        while np.all(self.point[:,-1] == 0.): 
            self.point = self.point[:,:-1] # Remove the z coordinates if this is a planar mesh
        self.gdim = self.point.shape[1]
        self.point_tag = np.zeros((self.point.shape[0],), dtype=np.uint32)

        # 2. read the higher-dimensional entities
        assert("gmsh:physical" in msh.cell_data)

        for cell, data in zip(msh.cells, msh.cell_data["gmsh:physical"]):
            if cell.type == "vertex":
                self.point_tag[cell.data] = data
            elif cell.type == "line":
                self.cell[1] = np.vstack((self.cell[1], np.hstack((cell.data, data[:, np.newaxis]))))
                if self.tdim < 1: 
                    self.tdim = 1
            elif cell.type == "triangle":
                self.cell[2] = np.vstack((self.cell[2], np.hstack((cell.data, data[:, np.newaxis]))))
                if self.tdim < 2:
                    self.tdim = 2
            else:
                raise RuntimeError("Unrecognized cell type. ")
            
    def get_entities(self, dim: int) -> dict:
        assert(dim > 0)
        if self.entities[dim] is None:
            if dim == 1:
                temp = self.cell[2][:, [0,1,1,2,2,0]].reshape(-1, 3, 2)
                edge_table = np.stack((np.min(temp, axis=2), np.max(temp, axis=2)), axis=2).reshape(-1, 2)
                edge_table = np.unique(edge_table, axis=0)
                self.edge_table = {r.tobytes(): i for i, r in enumerate(edge_table)}
            elif dim == 2:
                raise NotImplementedError
        return self.entities[dim]
    
    def draw(self) -> None:
        if self.tdim == 3:
            pass
        elif self.tdim == 2:
            if self.gdim == 2:
                pyplot.triplot(self.point[:,0], self.point[:,1], self.cell[2][:, :-1])
            elif self.gdim == 3:
                # use plot_trisurf
                raise NotImplementedError
        elif self.tdim == 1:
            pass
        pyplot.axis("equal")

