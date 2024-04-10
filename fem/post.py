from math import log2
import numpy as np
from .mesh import Mesh
from .funcspace import FunctionSpace

class NodeVisualizer:
    """
    Convert the nodal values to an array in align with the order of mesh.point, 
    and visualize them. 
    """

    mesh: Mesh
    fs: FunctionSpace
    nodal_repeats: int
    nodal_remap: np.ndarray

    def __init__(self, mesh: Mesh, fs: FunctionSpace) -> None:
        self.mesh = mesh
        self.fs = fs
        # 1. Find the unique mesh nodes from the mesh cells
        mesh_cells = mesh.cell[mesh.tdim].reshape(-1)
        _, point_remap = np.unique(mesh_cells, return_index=True) # (Ne * (tdim+1), )
        # 2. Find the nodal dofs in the function space
        nrp = len(fs.elem.dof_name[0])
        self.nodal_repeats = nrp
        elem_cells = fs.elem_dof[:(mesh.tdim+1)*nrp:nrp] # (tdim+1, Ne)
        elem_cells = elem_cells.T.reshape(-1) # (Ne * (tdim+1), )
        assert np.all(elem_cells % nrp == 0)
        elem_cells = elem_cells // nrp
        # 3. Build the remap
        self.nodal_remap = -np.ones((mesh.point.shape[0], ), dtype=np.int32)
        self.nodal_remap = elem_cells[point_remap]
        assert np.all(self.nodal_remap != -1)

    def remap(self, u: np.ndarray, num_copy: int = 1) -> np.ndarray:
        r = np.zeros((self.mesh.point.shape[0], num_copy))
        for i in range(num_copy):
            r[:,i] = u[self.nodal_remap*num_copy+i]
        return r

def printConvergenceTable(mesh_table, error_table) -> None:
    """ 
    Print the convergence table. 
    mesh_table: a list of string for mesh sizes as table headers. 
    error_table: a dict, "norm_type": [errors on each level].
    """
    m = len(mesh_table)
    # print the header
    header_str = "{0: <20}".format("")
    for i in range(m-1):
        header_str += "{0: <10}{1: <8}".format(mesh_table[i], "rate")
    header_str += "{0: <10}".format(mesh_table[-1])
    print(header_str)
    # print each norm
    for (norm_type, error_list) in error_table.items():
        error_str = "{0: <20}".format(norm_type)
        for i in range(m-1):
            error_str += "{0:<10.2e}{1:<8.2f}".format(error_list[i], log2(error_list[i]/error_list[i+1]))
        error_str += "{0:<10.2e}".format(error_list[-1])
        print(error_str)
    