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

    facet_ref: np.ndarray
    # mapping from facets to elements, (2, 2, num_facets = cell[tdim-1].shape[0])
    # [positive/negaive side, element id/facet id, *] 

    coord_fe: Any # type: FunctionSpace # the finite element space for the mesh mapping
    coord_map: Any # type: MeshMapping # the finite element function for the mesh mapping
    

    def __init__(self) -> None:
        self.gdim = 0
        self.tdim = 0
        self.point = None
        self.point_tag = None
        self.cell = [np.zeros((0, i+1), dtype=np.int32) for i in range(4)]
        self.cell_tag = [np.zeros((0, ), dtype=np.int32) for _ in range(4)]
        self.facet_ref = None
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

        for cell, data in zip(msh.cells, msh.cell_data["gmsh:physical"]):
            if cell.type == "vertex":
                self.point_tag[cell.data] = data
            elif cell.type == "line":
                self.cell[1] = np.vstack((self.cell[1], cell.data.astype(np.int32)))
                self.cell_tag[1] = np.concatenate((self.cell_tag[1], data.astype(np.int32)))
                self.tdim = max(self.tdim, 1)
            elif cell.type == "triangle":
                self.cell[2] = np.vstack((self.cell[2], cell.data.astype(np.int32)))
                self.cell_tag[2] = np.concatenate((self.cell_tag[2], data.astype(np.int32)))
                self.tdim = max(self.tdim, 2)
            else:
                raise RuntimeError("Unrecognized cell type. ")
        # assign the 0-th dim cells to be the tagged nodes
        self.cell[0] = np.nonzero(self.point_tag)[0].reshape(-1, 1)
        self.cell_tag[0] = self.point_tag[self.cell[0]].reshape(-1)

        # 3. build the facets. 
        self.build_facet_ref()

    def build_facet_ref(self) -> None:
        """
        This function will tag all the untagged facets with ID 99. 
        """
        if self.tdim == 0:
            return
        # collect all the facets
        all_facets = ref_doms[self.tdim]._get_sub_entities(self.cell[self.tdim], dim=self.tdim-1) # (Ne, num_sub_ent, tdim-1+1)
        num_facet = all_facets.shape[1]
        all_facets = all_facets.reshape(-1, self.tdim)
        all_facets.sort(axis=1) # maybe need manual sort for better performance
        uq_facets, idx = np.unique(all_facets, return_index=True, axis=0)
        tagged_facets = np.sort(self.cell[self.tdim-1], axis=1)
        sub_idx = binsearchkw(uq_facets.astype(np.int32), tagged_facets)
        assert np.all(sub_idx != -1)
        # the first side 
        Nf = uq_facets.shape[0]
        self.facet_ref = np.zeros((2, 2, Nf), dtype=np.int32)
        self.facet_ref[0,0], self.facet_ref[0,1] = np.divmod(idx, num_facet)
        # the other side
        _, idx = np.unique(all_facets[::-1], return_index=True, axis=0)
        idx = all_facets.shape[0] - idx - 1
        self.facet_ref[1,0], self.facet_ref[1,1] = np.divmod(idx, num_facet)
        # save all the facets
        self.cell[self.tdim-1] = uq_facets
        old_tags = self.cell_tag[self.tdim-1]
        self.cell_tag[self.tdim-1] = 99 * np.ones((Nf, ), dtype=np.int32)
        self.cell_tag[self.tdim-1][sub_idx] = old_tags
        # fix the facet orientation
        tags = self.cell_tag[self.tdim][self.facet_ref[:,0]] # (2, num_facet)
        flipped = tags[0] > tags[1]
        self.facet_ref[:,:,flipped] = self.facet_ref[::-1,:,flipped]
    
    def view(self, dim: int, sub_ids: Optional[tuple[int]] = None) -> "Mesh":
        submesh = Mesh()
        submesh.tdim = dim
        submesh.gdim = self.gdim
        keep_idx = [None] * (dim+1)
        # 1. select the entities of the highest dimension to preserve
        elem_tag = self.point_tag if dim == 0 else self.cell_tag[dim]
        keep_idx[dim] = np.zeros((elem_tag.shape[0], ), dtype=np.bool_)
        if sub_ids == None:
            keep_idx = True
        else:
            assert isinstance(sub_ids, tuple)
            for t in sub_ids:
                keep_idx[dim][elem_tag == t] = True
        if dim == 0:
            submesh.point = self.point[keep_idx[0]]
            submesh.point_tag = self.point_tag[keep_idx[0]]
            return submesh
        submesh.cell[dim] = self.cell[dim][keep_idx[dim]]
        submesh.cell_tag[dim] = self.cell_tag[dim][keep_idx[dim]]
        # 2. Select the nodes to preserve and construct the node remap
        keep_idx[0] = np.unique(submesh.cell[dim].reshape(-1))
        submesh.point = self.point[keep_idx[0]]
        submesh.point_tag = self.point_tag[keep_idx[0]]
        point_remap = -np.ones((self.point.shape[0],), dtype=np.int32)
        point_remap[keep_idx[0]] = np.arange(submesh.point.shape[0], dtype=np.int32)
        # 3. Collect the entites of lower dimensions
        for d in range(0, dim):
            submesh.cell[d] = point_remap[self.cell[d]]
            keep_idx[d] = np.all(submesh.cell[d] != -1, axis=1)
            submesh.cell[d] = submesh.cell[d][keep_idx[d]]
            submesh.cell_tag[d] = self.cell_tag[d][keep_idx[d]]
        # 3. Remap the elements
        submesh.cell[dim] = point_remap[submesh.cell[dim]]
        # 4. 
        submesh.build_facet_ref()
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

