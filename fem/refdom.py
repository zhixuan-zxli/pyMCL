from typing import Any
import numpy as np

class RefCell:
    tdim: int
    dx: float
    vertices: np.ndarray
    sub_entities: Any

class RefNode(RefCell):
    tdim: int = 0
    dx: float = 1.0
    vertices: np.ndarray = np.array((0.0,))
    sub_entities = tuple()

class RefLine(RefCell):
    tdim: int = 1
    dx: float = 1.0
    vertices: np.ndarray = np.array(
        ((0.0,), (1.0,))
    )
    sub_entities = (
        None, # nodes
    )

class RefTri(RefCell):
    tdim: int = 2
    dx: float = 1.0/2
    vertices: np.ndarray = np.array(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    )
    sub_entities = (
        None, # nodes
        np.array(((0,1), (1,2), (2,0)), dtype=np.int32) # edges
    )

# the collection of all the reference domains by dimension
ref_doms = (RefNode, RefLine, RefTri, None)
