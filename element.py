import numpy as np

class Element:
    def __init__(self, type: str, cell: str, degree: int) -> None:
        self.type = type
        self.cell = cell
        self.degree = degree
        self.__basis_table = dict()

    def __str__(self) -> str:
        return "Degree-{} {} element on {}".format(self.degree, self.type, self.cell)
    
    def basis(self, qpts: np.ndarray) -> np.ndarray:
        return self.__basis_table[self.degree](qpts)
    
class Lagrange(Element):
    def __init__(self, cell: str, degree: int) -> None:
        super().__init__("Lagrange", cell, degree)
        if cell == "edge":
            self.__basis_table = {1: self.__basis_P1D1, 2: self.__basis_P2D1}
        elif cell == "tri":
            self.__basis_table = {1: self.__basis_P1D2, 2: self.__basis_P2D2}
        else:
            raise RuntimeError("Cell type: {} not implemented. ".format(cell))

    def __basis_P1D1(qpts: np.ndarray) -> np.ndarray:
        return qpts
    
    def __basis_P2D1(qpts: np.ndarray) -> np.ndarray:
        return qpts
    
    def __basis_P1D2(qpts: np.ndarray) -> np.ndarray:
        return qpts
    
    def __basis_P2D2(qpts: np.ndarray) -> np.ndarray:
        return qpts