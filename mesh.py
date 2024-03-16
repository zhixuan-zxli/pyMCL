import meshio
import numpy as np
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
            
    def add_constraint(self, master_marker, slave_marker, transform, tol: float = 1e-14) -> None:
        master_idx = np.nonzero(master_marker(self.point))[0] # master indices
        master_data = np.hstack((transform(self.point[master_idx]), master_idx[:,np.newaxis]))
        slave_idx = np.nonzero(slave_marker(self.point))[0] # slave indices
        slave_data = np.hstack((self.point[slave_idx], -slave_idx[:,np.newaxis]-1))
        assert master_data.shape[0] == slave_data.shape[0], "Number of nodes unmatched. "
        data = np.vstack((master_data, slave_data))
        si = np.lexsort((data[:,2], data[:,1], data[:,0]))
        data = data[si, :]
        error = data[::2, :2] - data[1::2, :2]
        assert np.linalg.norm(error.reshape(-1), ord=np.inf) < tol
        con = data[:, 2].astype(np.int32).reshape(-1, 2)
        con = np.vstack((np.min(con, axis=1), np.max(con, axis=1))).T #(x,2)
        con[:,0] = -con[:,0] - 1
        if not hasattr(self, "constraint_table"):
            self.constraint_table = []
        self.constraint_table.append(con)

    def _get_point_remap(self) -> None:
        """
        Link the slave points to the master points. 
        Ensure that there is no intermediate slave poitns. 
        """
        assert hasattr(self, "constraint_table"), "Why get point remap if there is no constraint?"
        if not hasattr(self, "point_remap"):
            con = np.vstack(self.constraint_table)
            remap = np.arange(self.point.shape[0], dtype=np.uint32)
            p_update = remap.copy()
            while True:
                p_update[con[:,0]] = remap[con[:,1]]
                if np.all(p_update == remap):
                    break
                remap[:] = p_update
            self.point_remap = remap
        return self.point_remap
    
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

