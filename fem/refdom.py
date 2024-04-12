import numpy as np

class RefCell:
    tdim: int
    dx: float
    facet_normal: np.ndarray # (num_facet, tdim)

    @staticmethod
    def _get_sub_entities(elem_cell: np.ndarray, dim: int) -> np.ndarray:
        """
        elem_cell: (Ne, d+1)
        return: (Ne, num_sub_entity, dim+1)
        """
        raise NotImplementedError

class RefNode(RefCell):
    tdim: int = 0
    dx: float = 1.0
    facet_normal: np.ndarray = np.ones((0,0))

    @staticmethod
    def _get_sub_entities(elem_cell: np.ndarray, dim: int) -> np.ndarray:
        assert elem_cell.shape[1] == 1
        if dim == 0:
            return elem_cell.reshape(-1, 1, 1)
        raise RuntimeError("Incorrect dimension for getting sub entities. ")

class RefLine(RefCell):
    tdim: int = 1
    dx: float = 1.0
    facet_normal: np.ndarray = np.array(((-1.0,), (1.0,),))

    @staticmethod
    def _get_sub_entities(elem_cell: np.ndarray, dim: int) -> np.ndarray:
        assert elem_cell.shape[1] == 2
        if dim == 0:
            return elem_cell.reshape(-1, 2, 1)
        elif dim == 1:
            return elem_cell.reshape(-1, 1, 2)
        raise RuntimeError("Incorrect dimension for getting sub entities. ")


class RefTri(RefCell):
    tdim: int = 2
    dx: float = 1.0/2
    facet_normal: np.ndarray = np.array(
        ((0.0, -1.0), 
         (0.7071067811865475, 0.7071067811865475), 
         (-1.0, 0.0))
    )

    @staticmethod
    def _get_sub_entities(elem_cell: np.ndarray, dim: int) -> np.ndarray:
        assert elem_cell.shape[1] == 3
        if dim == 0:
            return elem_cell.reshape(-1, 3, 1)
        elif dim == 1:
            return elem_cell[:, [0,1,1,2,2,0]].reshape(-1, 3, 2)
        elif dim == 2:
            return elem_cell.reshape(-1, 1, 3)
        raise RuntimeError("Incorrect dimension for getting sub entities.")

# the collection of all the reference domains by dimension
ref_doms = (RefNode, RefLine, RefTri)
