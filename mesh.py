from typing import Optional
import meshio
import numpy as np
from scipy.sparse import csr_array
from matplotlib import pyplot

class Measure:
    def __init__(self, tdim: int, sub_id: Optional[tuple[int]] = None) -> None:
        self.tdim = tdim
        self.sub_id = sub_id

class Mesh:

    gdim: int
    tdim: int
    point: np.ndarray
    point_tag: np.ndarray
    cell: list[np.ndarray]
    # coord_fe: "FiniteElement"
    # coord_map: "Function"

    def __init__(self) -> None:
        self.gdim = 0
        self.tdim = 0
        self.point = None
        self.point_tag = None
        self.cell = [
            None, 
            np.zeros((0, 3), dtype=np.uint32), # edge
            np.zeros((0, 4), dtype=np.uint32), # triangle
            np.zeros((0, 5), dtype=np.uint32)                    
        ]

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
        master_idx = np.nonzero(master_marker(self.point))[0] if callable(master_marker) else master_marker # master indices
        master_data = np.hstack((transform(self.point[master_idx]), master_idx[:,np.newaxis]))
        slave_idx = np.nonzero(slave_marker(self.point))[0] if callable(slave_marker) else slave_marker # slave indices
        slave_data = np.hstack((self.point[slave_idx], -slave_idx[:,np.newaxis]-1))
        assert master_data.shape[0] == slave_data.shape[0], "Number of nodes unmatched. "
        # try to match the slave/master pair by sorting
        data = np.vstack((master_data, slave_data))
        si = np.lexsort((data[:,2], data[:,1], data[:,0]))
        data = data[si, :]
        error = data[::2, :2] - data[1::2, :2]
        assert np.linalg.norm(error.reshape(-1), ord=np.inf) < tol
        # save the matching pairs
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
            temp = remap.copy()
            while True:
                temp[con[:,0]] = remap[con[:,1]]
                if np.all(temp == remap):
                    break
                remap[:] = temp
            self.point_remap = remap
        return self.point_remap
    
    @staticmethod
    def _get_edges_from_tri(Np, tri):
        """
        Build the edge map e -> id, 
        where e = (p1, p2) with p1 < p2 represents an edge, 
        and id is 1-based and unique among the edges. 
        Return: m: csr_array, edges: (Nt*3, 2)
        """
        Nt = tri.shape[0]
        edges = tri[:, [0,1,1,2,2,0]].reshape(-1, 3, 2)
        edges = np.stack((np.min(edges, axis=2), np.max(edges, axis=2)), axis=2).reshape(-1, 2)
        m = csr_array((np.ones((Nt*3,), dtype=np.int64), (edges[:,0], edges[:,1])), shape=(Np, Np))
        m.data = np.arange(m.nnz) + 1
        return m, edges
    
    def view(self, mea: Measure) -> "Mesh":
        submesh = Mesh()
        submesh.tdim = mea.tdim
        submesh.gdim = self.gdim
        # select the entities to preserve
        keep_idx = [None] * 4
        if mea.tdim == 3:
            raise NotImplementedError
        elif mea.tdim == 2:
            flag = np.zeros((self.cell[2].shape[0], ), dtype=np.bool8)
            for t in mea.sub_id:
                flag[self.cell[2][:,-1] == t] = True
            keep_idx[2] = np.nonzero(flag)[0]
            # find out the edges to be preserved
            edge_map, _ = self._get_edges_from_tri(self.point.shape[0], self.cell[2][keep_idx[2], :])
            edge_flag = edge_map[np.min(self.cell[1][:, :-1], axis=1), np.max(self.cell[1][:, :-1])] > 0
            keep_idx[1] = np.nonzero(edge_flag)[0]
            # find out the vertices to be preserved
            keep_idx[0] = np.unique(self.cell[2][keep_idx[2], :-1])
        elif mea.tdim == 1:
            flag = np.zeros((self.cell[1].shape[0], ), dtype=np.bool8)
            for t in mea.sub_id:
                flag[self.cell[1][:, -1] == t] = True
            keep_idx[1] = np.nonzero(flag)[0]
            keep_idx[0] = np.unique(self.cell[1][keep_idx[1], :-1])
        elif mea.tdim == 0:
            flag = np.zeros((self.point.shape[0], ), dtype=np.bool8)
            for t in mea.sub_id:
                flag[self.point_tag == t] = True
            keep_idx[0] = np.nonzero(flag)[0]
        # copy the nodes
        submesh.point = self.point[keep_idx[0]]
        submesh.point_tag = self.point_tag[keep_idx[0]]
        submesh.parent_point = keep_idx[0]
        Np = submesh.point.shape[0]
        # remap the nodes
        point_remap = np.zeros((self.point.shape[0], ), dtype=np.uint32)
        point_remap[keep_idx[0]] = np.arange(Np)
        for d in (1,2,3):
            if keep_idx[d] is not None and keep_idx[d].size > 0:
                submesh.cell[d] = np.zeros((keep_idx[d].shape[0], self.cell[d].shape[1]), dtype=np.uint32)
                submesh.cell[d][:, :-1] = point_remap[self.cell[d][keep_idx[d], :-1]]
                submesh.cell[d][:, -1] = self.cell[d][keep_idx[d], -1]
        return submesh
    
    def draw(self) -> None:
        if self.tdim == 3:
            print("Unable to visualize 3D mesh. ")
        elif self.tdim == 2:
            if self.gdim == 2:
                pyplot.triplot(self.point[:,0], self.point[:,1], triangles=self.cell[2][:, :-1])
            elif self.gdim == 3:
                # use plot_trisurf
                raise NotImplementedError
        elif self.tdim == 1:
            if self.gdim ==  2:
                from matplotlib.collections import LineCollection
                p0 = self.point[self.cell[1][:,0]]
                p1 = self.point[self.cell[1][:,1]]
                segs = np.concatenate((p0[:,np.newaxis], p1[:,np.newaxis]), axis=1) # (nseg x 2 x 2)
                pyplot.gca().add_collection(LineCollection(segments=segs))
            else:
                raise NotImplementedError
        pyplot.axis("equal")

