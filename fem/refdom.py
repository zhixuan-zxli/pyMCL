from typing import Any
import numpy as np

class RefCell:
    tdim: int
    dx: float
    ds: np.ndarray # (num_facet,)
    vertices: np.ndarray
    facet_normal: np.ndarray # (tdim, num_facet)
    sub_entities: Any # todo: may turn sub_entities into a function for efficiency

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
    sub_entities = tuple()

class RefLine(RefCell):
    tdim: int = 1
    dx: float = 1.0
    vertices: np.ndarray = np.array(
        ((0.0,), (1.0,))
    )
    ds: np.ndarray = np.array((1.0, 1.0))
    facet_normal: np.ndarray = np.array(((-1.0, 1.0),))
    sub_entities = (
        np.array(((0,), (1,)), dtype=np.int32), # nodes
    )

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
        ((0.0, 0.7071067811865475, -1.0), 
        (-1.0, 0.7071067811865475, 0.0), )
    )
    sub_entities = (
        np.array(((0,), (1,), (2,)), dtype=np.int32), # nodes
        np.array(((0,1), (1,2), (2,0)), dtype=np.int32) # edges
    )

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
ref_doms = (RefNode, RefLine, RefTri, None)
