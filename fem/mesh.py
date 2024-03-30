from typing import Optional, Any
import meshio
import numpy as np
from .refdom import ref_doms
from tools.binsearchkw import binsearchkw
from matplotlib import pyplot

class Mesh:

    gdim: int
    tdim: int
    point: np.ndarray
    point_tag: np.ndarray
    cell: list[np.ndarray]
    # cell[d] stores the node representation of a d-dimensional simplex, appended by a tag. 
    cell_tag: list[np.ndarray]

    cell_entity: list[np.ndarray]
    # cell_entity[d] is the d-dimensional sub-entity of a cell of tdim dimensions. 

    inv_bdry: np.ndarray

    # entity_list: list[np.ndarray]
    # # entity[d] stores the d-dimensional entities in a N-by-(d+1) array, 
    # # where N is the number of entities and each row contains the nodes of one entity. 
    # entity_tag: list[np.ndarray]
    # # entity_tag[d] is just a translation from cell[d]. 
    # # It is an N-by-2 array, each row being the ID of the entity and the tag. 

    coord_fe: Any  # the finite element space for the mesh mapping
    coord_map: Any # the finite element function for the mesh mapping
    

    def __init__(self) -> None:
        self.gdim = 0
        self.tdim = 0
        self.point = None
        self.point_tag = None
        self.coord_fe = None
        self.coord_map = None

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
        self.point_tag = np.zeros((self.point.shape[0],), dtype=np.int32)

        # 2. read the higher-dimensional entities
        assert("gmsh:physical" in msh.cell_data)

        dim_map = {"line": 1, "triangle": 2}

        mesh_cell = [
            None, 
            np.zeros((0, 2), dtype=np.int32), 
            np.zeros((0, 3), dtype=np.int32)
        ]

        mesh_tag = [
            None, 
            np.zeros((0,), dtype=np.int32), 
            np.zeros((0,), dtype=np.int32)
        ]

        for cell, data in zip(msh.cells, msh.cell_data["gmsh:physical"]):
            cell.data = cell.data.astype(np.int32)
            data = data.astype(np.int32)
            if cell.type == "vertex":
                self.point_tag[cell.data] = data
            else:
                d = dim_map[cell.type]
                mesh_cell[d] = np.vstack((mesh_cell[d], cell.data))
                mesh_tag[d] = np.concatenate((mesh_tag[d], data))
                self.tdim = max(self.tdim, d)

        # 3. interpret the mesh data
        self.cell, self.cell_entity, self.inv_bdry = self.build_cells(mesh_cell[self.tdim])
        self.cell.append(mesh_cell[self.tdim])
        # reset the tags
        self.cell_tag = [None] * (self.tdim+1)
        for d in range(1, self.tdim):
            mesh_cell[d].sort(axis=1)
            idx = binsearchkw(self.cell[d], mesh_cell[d])
            assert np.all(idx >= 0)
            self.cell_tag[d] = np.zeros((self.cell[d].shape[0],), dtype=np.int32)
            self.cell_tag[d][idx] = mesh_tag[d]
        self.cell_tag[self.tdim] = mesh_tag[self.tdim]

    @staticmethod
    def build_cells(elem_cell: np.ndarray):
        """
        elem_cell: the list of cells of the highest topological dimension in this mesh. 
        """
        tdim = elem_cell.shape[1]-1
        assert tdim >= 1
        ref_cell = ref_doms[tdim]
        num_cell = elem_cell.shape[0]
        # prepare the output
        cell = [None] * tdim
        cell_entity = [None] * tdim
        # collect the entities from dimension 0, ..., tdim-1
        for d in range(tdim-1, -1, -1):
            if d > 0:
                sub_ent = ref_cell.sub_entities[d]
                all_entities = elem_cell[:, sub_ent.ravel()].reshape(-1, sub_ent.shape[1])
                all_entities.sort(axis=1)
            else:
                all_entities = elem_cell.ravel()
            cell[d], idx, inv_idx = np.unique(all_entities, return_index=True, return_inverse=True, axis=0)
            cell_entity[d] = inv_idx.reshape(num_cell, -1).astype(np.int32) # (num_cell, tdim+1)
            if d == tdim-1: # get the inverse of the boundary map for co-dimension one entity   
                inv_bdry = np.zeros((2, 2, cell[d].shape[0]), dtype=np.int32) #[positive/negaive side, element id/facet id, *]       
                inv_bdry[0,0], inv_bdry[0,1] = np.divmod(idx, sub_ent.shape[0])
                _, idx = np.unique(all_entities[::-1], return_index=True, axis=0) # find the second occurences
                idx = all_entities.shape[0] - idx - 1
                inv_bdry[1,0], inv_bdry[1,1] = np.divmod(idx, sub_ent.shape[0])
        return cell, cell_entity, inv_bdry
    
    # set boundary orientation ...
    
    def view(self, dim: int, sub_ids: Optional[tuple[int]] = None) -> "Mesh":
        submesh = Mesh()
        submesh.tdim = dim
        submesh.gdim = self.gdim
        if dim == 0:
            keep_idx = np.zeros((self.point_tag.shape[0],), dtype=np.bool8)
            if sub_ids == None:
                keep_idx = True
            else:
                assert isinstance(sub_ids, tuple)
                for t in sub_ids:
                    keep_idx[self.point_tag == t] = True
            submesh.point = self.point[keep_idx]
            submesh.point_tag = self.point_tag[keep_idx]
            return submesh
        # 1. select the entities of the highest dimension to preserve
        keep_idx = np.zeros((self.cell_tag[dim].shape[0],), dtype=np.bool8)
        if sub_ids == None:
            keep_idx = True
        else:
            assert isinstance(sub_ids, tuple)
            for t in sub_ids:
                keep_idx[self.cell_tag[dim] == t] = True
        elem_cell = self.cell[dim][keep_idx]
        # 2. Collect the entites of lower dimensions
        submesh.cell, submesh.cell_entity, submesh.inv_bdry = self.build_cells(elem_cell)
        submesh.cell.append(elem_cell)
        # 3. Calculate the remap for the nodes.
        submesh.point = self.point[submesh.cell[0]]
        submesh.point_tag = self.point_tag[submesh.cell[0]]
        point_remap = -np.ones((self.point.shape[0],), dtype=np.int32)
        point_remap[submesh.cell[0]] = np.arange(submesh.point.shape[0], dtype=np.int32)
        # 4. Reset the cell tags. xxx
        submesh.cell_tag = [None] * (dim+1)
        for d in range(1, dim):
            valid_tag = self.cell_tag[d][self.cell_tag[d] != 0]
            tagged_cell = self.cell[d][self.cell_tag[d] != 0]
            idx = binsearchkw(submesh.cell[d], tagged_cell)
            submesh.cell_tag[d] = np.zeros((submesh.cell[d].shape[0],), dtype=np.int32)
            submesh.cell_tag[d][idx[idx >= 0]] = valid_tag[idx >= 0]
        submesh.cell_tag[dim] = self.cell_tag[dim][keep_idx]
        # 4. Remap the nodes. 
        for d in range(1, dim+1):
            submesh.cell[d] = point_remap[submesh.cell[d]]
            assert np.all(submesh.cell[d] >= 0)
        if dim == 1:
            submesh.inv_bdry = submesh.inv_bdry[:,:,submesh.cell[0]]
        # submesh.cell[0] = None
        return submesh

    # def add_constraint(self, master_marker, slave_marker, transform, tol: float = 1e-14) -> None:
    #     master_idx = np.nonzero(master_marker(self.point))[0] if callable(master_marker) else master_marker # master indices
    #     master_data = np.hstack((transform(self.point[master_idx]), master_idx[:,np.newaxis]))
    #     slave_idx = np.nonzero(slave_marker(self.point))[0] if callable(slave_marker) else slave_marker # slave indices
    #     slave_data = np.hstack((self.point[slave_idx], -slave_idx[:,np.newaxis]-1))
    #     assert master_data.shape[0] == slave_data.shape[0], "Number of nodes unmatched. "
    #     # try to match the slave/master pair by sorting
    #     data = np.vstack((master_data, slave_data))
    #     si = np.lexsort((data[:,2], data[:,1], data[:,0]))
    #     data = data[si, :]
    #     error = data[::2, :2] - data[1::2, :2]
    #     assert np.linalg.norm(error.reshape(-1), ord=np.inf) < tol
    #     # save the matching pairs
    #     con = data[:, 2].astype(np.int32).reshape(-1, 2)
    #     con = np.vstack((np.min(con, axis=1), np.max(con, axis=1))).T #(x,2)
    #     con[:,0] = -con[:,0] - 1
    #     if not hasattr(self, "constraint_table"):
    #         self.constraint_table = []
    #     self.constraint_table.append(con)    
    
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

