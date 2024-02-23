
class Element:
    def __init__(self, type: str, degree: int) -> None:
        self.type = type
        self.degree = degree

    def __str__(self) -> str:
        return "Degree-{} {} element".format(self.degree, self.type)
    
class Lagrange1(Element):
    def __init__(self) -> None:
        super().__init__("Lagrange", 1)

class Lagrange2(Element):
    def __init__(self) -> None:
        super().__init__("Lagrange", 2)
