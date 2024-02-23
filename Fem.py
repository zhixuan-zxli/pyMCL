from Mesh import Mesh
import Element

class FunctionSpace:
    def __init__(self, mesh: Mesh, elem: Element, vecdim: int = 1) -> None:
        self.mesh = mesh
        self.elem = elem
        self.vecdim = vecdim
