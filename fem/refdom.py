from typing import Any
import numpy as np

class RefCell:
    tdim: int
    dx: float
    ds: np.ndarray # (num_facet,)
    vertices: np.ndarray
    facet_normal: np.ndarray # (num_facet, tdim)

    @staticmethod
    def _get_sub_entities(elem_cell: np.ndarray, dim: int) -> np.ndarray:
        """
        elem_cell: (Ne, d+1)
        return: (Ne, num_sub_entity, dim+1)
        """
        raise NotImplementedError

    @staticmethod
    def _broadcast_facet(quad_pts: np.ndarray) -> np.ndarray:
        """
        quad_pts: (Nd, Nq)
        return: (Nd, num_facet, Nq)
        """
        # implementation by subclass
        raise NotImplementedError

class RefNode(RefCell):
    tdim: int = 0
    dx: float = 1.0
    vertices: np.ndarray = np.array((0.0,))
    ds: np.ndarray = np.ones((0,))
    facet_normal: np.ndarray = np.ones((0,0))

class RefLine(RefCell):
    tdim: int = 1
    dx: float = 1.0
    vertices: np.ndarray = np.array(
        ((0.0,), (1.0,))
    )
    ds: np.ndarray = np.array((1.0, 1.0))
    facet_normal: np.ndarray = np.array(((-1.0,), (1.0,),))

    @staticmethod
    def _get_sub_entities(elem_cell: np.ndarray, dim: int) -> np.ndarray:
        assert elem_cell.shape[1] == 2
        if dim == 0:
            return elem_cell.reshape(-1, 2, 1)
        elif dim == 1:
            return elem_cell
        raise RuntimeError("Incorrect dimension for getting sub entities. ")

    @staticmethod
    def _broadcast_facet(quad_pts: np.ndarray) -> np.ndarray:
        # quad_pts: (1, 1)
        # return: (1, 2, 1)
        r = np.zeros((1,2,1))
        r[0,1,0] = 1.0
        return r


class RefTri(RefCell):
    tdim: int = 2
    dx: float = 1.0/2
    vertices: np.ndarray = np.array(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    )
    ds: np.ndarray = np.array((1.0, 1.4142135623730951, 1.0))
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
            return elem_cell
        raise RuntimeError("Incorrect dimension for getting sub entities.")


    @staticmethod
    def _broadcast_facet(quad_pts: np.ndarray) -> np.ndarray:
        # quad_pts: (1, Nq)
        # return: (2, 3, Nq)
        r = np.zeros((2, 3, quad_pts.shape[1]))
        r[0,0] = quad_pts
        r[0,1] = 1.0 - quad_pts
        r[1,1] = quad_pts
        r[1,2] = 1.0 - quad_pts

# the collection of all the reference domains by dimension
ref_doms = (RefNode, RefLine, RefTri)
