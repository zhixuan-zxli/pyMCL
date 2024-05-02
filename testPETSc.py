import numpy as np
from scipy.sparse import load_npz, csr_array
import petsc4py
from petsc4py.PETSc import Vec, Mat, NormType, KSP, PC
from timeit import default_timer

if __name__ == "__main__":
    meta = np.load("./A_meta.npz")
    free_dof = meta["free_dof"]
    num_dof_U = meta["num_dof_U"].item()
    L = meta["L"]
    A = load_npz("./A.npz") # type: csr_array

    petsc4py.init()
    L_Vec = Vec().createWithArray(L)
    A_Mat = Mat().createAIJ(A.shape, csr=(A.indptr, A.indices))
    A_Mat.setValuesCSR(A.indptr, A.indices, A.data)
    A_Mat.assemblyBegin()
    A_Mat.assemblyEnd()
    print("1-norm of A = {}".format(A_Mat.norm(NormType.NORM_1)))
    
    ksp = KSP().create()
    ksp.setType(KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PC.Type.LU)
    pc.setFactorSolverType("umfpack")
    ksp.setOperators(A_Mat)
    Z_Vec = A_Mat.createVecRight()
    start_time = default_timer()
    ksp.solve(L_Vec, Z_Vec)
    print("UMFPACK = {}s".format(default_timer() - start_time))
    z = Z_Vec.getArray()
    
