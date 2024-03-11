import numpy as np
from fe import RefCell, RefNode, RefLine, RefTri

class Quadrature:

    _node = np.array(
        ((0.0, 1.0))
    ).T

    _line_O4 = np.array(
        ((0.21132486540518708, 1.0/2), 
         (0.7886751345948129, 1.0/2))
    ).T

    _tri_O3 = np.array(
        ((1.0/2, 0.0, 1.0/2, 1.0/3),  
        (1.0/2, 1.0/2, 0.0, 1.0/3), 
        (0.0, 1.0/2, 1.0/2, 1.0/3))
    ).T
    _tri_O4 = np.array(
      ((0.445948490915965, 0.108103018168070, 0.445948490915965, 0.223381589678011), 
      (0.445948490915965, 0.445948490915965, 0.108103018168070, 0.223381589678011), 
      (0.108103018168070, 0.445948490915965, 0.445948490915965, 0.223381589678011),
      (0.091576213509771, 0.816847572980459, 0.091576213509771, 0.109951743655322), 
      (0.091576213509771, 0.091576213509771, 0.816847572980459, 0.109951743655322),
      (0.816847572980459, 0.091576213509771, 0.091576213509771, 0.109951743655322))
    ).T
    
    @staticmethod
    def getTable(cellType: type, order: int) -> np.array:
        if cellType == RefNode:
            print("Retriving quadrature table for node. ")
            return Quadrature._node
        if cellType == RefLine:
            if order <= 4:
                return Quadrature._line_O4
        if cellType == RefTri:
            if order <= 3:
                return Quadrature._tri_O3
            if order == 4:
                return Quadrature._tri_O4
        raise NotImplementedError
