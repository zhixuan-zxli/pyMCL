import meshio
import numpy as np
from scipy.sparse import csr_array
from matplotlib import pyplot

class Mesh:

    gdim: int
    tdim: int
    point: np.ndarray
    point_tag: np.ndarray
    # cell
    # entities
    # coord_fe: "FiniteElement"
    # coord_map: "Function"

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
            
        # 3. read periodicity. 
        if hasattr(msh, "gmsh_periodic"):
            corr = [item[3] for item in msh.gmsh_periodic if item[0] == self.tdim-1]
            corr = np.vstack(corr)
            p_map = np.arange(self.point.shape[0], dtype=np.uint32)
            p_update = p_map.copy()
            while True:
                p_update[corr[:,0]] = p_map[corr[:,1]]
                if np.all(p_update == p_map):
                    break
                p_map[:] = p_update
            self.p_map = p_map
            
    def get_entities(self, dim: int) -> dict:
        assert(dim > 0)
        if self.entities[dim] is None:
            if dim == 1:
                Np, Nt = self.point.shape[0], self.cell[2].shape[0]
                idx = self.cell[2][:, [0,1,1,2,2,0]].reshape(-1, 3, 2)
                idx = np.stack((np.min(idx, axis=2), np.max(idx, axis=2)), axis=2).reshape(-1, 2)
                m = csr_array((np.ones((Nt*3,), dtype=np.int64), (idx[:,0], idx[:,1])), shape=(Np, Np))
                m.data = np.arange(m.nnz) + 1
                self.entities[1] = m
            elif dim == 2:
                raise NotImplementedError
        return self.entities[dim]
    
    def draw(self) -> None:
        if self.tdim == 3:
            print("Unable to visualize 3D mesh. ")
        elif self.tdim == 2:
            if self.gdim == 2:
                pyplot.triplot(self.point[:,0], self.point[:,1], self.cell[2][:, :-1])
            elif self.gdim == 3:
                # use plot_trisurf
                raise NotImplementedError
        elif self.tdim == 1:
            raise NotImplementedError
        pyplot.axis("equal")

