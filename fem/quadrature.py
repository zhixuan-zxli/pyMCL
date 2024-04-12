import numpy as np
from .refdom import RefNode, RefLine, RefTri

class Quadrature:

    _node = (
        np.array((0.,)), 
        np.array((1., )), 
    )

    _line_O4 = (
        np.array(((0.21132486540518708, 0.7886751345948129), 
                  (0.7886751345948129, 0.21132486540518708))), 
        np.array((1.0/2, 1.0/2)),
    )
    _line_O6 = (
        np.array(((0.1127016653792583, 0.8872983346207417), 
                  (0.5, 0.5), 
                  (0.8872983346207417, 0.1127016653792583))), 
        np.array((5.0/18, 4.0/9, 5.0/18)), 
    )

    _tri_O3 = (
        np.array(((1.0/2, 0.0, 1.0/2), 
                  (1.0/2, 1.0/2, 0.0), 
                  (0.0, 1.0/2, 1.0/2))), 
        np.array((1.0/3, 1.0/3, 1.0/3)), 
    )

    _tri_O4 = (
        np.array(((0.445948490915965, 0.108103018168070, 0.445948490915965), 
                  (0.445948490915965, 0.445948490915965, 0.108103018168070), 
                  (0.108103018168070, 0.445948490915965, 0.445948490915965),
                  (0.091576213509771, 0.816847572980459, 0.091576213509771), 
                  (0.091576213509771, 0.091576213509771, 0.816847572980459),
                  (0.816847572980459, 0.091576213509771, 0.091576213509771))), 
        np.array((0.223381589678011, 0.223381589678011, 0.223381589678011, 
                  0.109951743655322, 0.109951743655322, 0.109951743655322)), 
    )
    
    @staticmethod
    def getTable(cellType: type, order: int) -> tuple[np.array, np.ndarray]:
        if cellType == RefNode:
            return Quadrature._node
        if cellType == RefLine:
            if order <= 4:
                return Quadrature._line_O4
            if order <= 6:
                return Quadrature._line_O6
        if cellType == RefTri:
            if order <= 3:
                return Quadrature._tri_O3
            if order == 4:
                return Quadrature._tri_O4
        raise NotImplementedError
