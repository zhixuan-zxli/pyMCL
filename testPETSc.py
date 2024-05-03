from sys import argv
import petsc4py
petsc4py.init(argv)
import numpy as np
from scipy.sparse import load_npz, csr_array
from petsc4py.PETSc import Vec, Mat, PC, KSP #, IS, NormType
from timeit import default_timer

if __name__ == "__main__":
    meta = np.load("./A_meta.npz")
    free_dof = meta["free_dof"]
    num_dof_U = meta["num_dof_U"].item()
    L = meta["L"]
    A = load_npz("./A.npz") # type: csr_array

    # extract the free part
    A = A[free_dof][:,free_dof]
    L = L[free_dof]
    num_dof_U = free_dof[:num_dof_U].sum()
    index_0 = np.arange(num_dof_U, dtype=np.int32)
    index_1 = np.arange(num_dof_U, A.shape[0], dtype=np.int32)

    L_Vec = Vec().createWithArray(L)
    A_Mat = Mat().createAIJ(A.shape, csr=(A.indptr, A.indices))
    A_Mat.setValuesCSR(A.indptr, A.indices, A.data)
    A_Mat.assemblyBegin()
    A_Mat.assemblyEnd()
    # print("1-norm of A = {}".format(A_Mat.norm(NormType.NORM_1)))
    
    # invoke umfpack
    ksp = KSP().create()
    ksp.setType(KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PC.Type.LU)
    pc.setFactorSolverType("umfpack")
    ksp.setOperators(A_Mat)
    ksp.setFromOptions()
    Z_Vec = A_Mat.createVecRight()
    start_time = default_timer()
    ksp.solve(L_Vec, Z_Vec)
    print("UMFPACK = {}s".format(default_timer() - start_time))
    z = Z_Vec.getArray()

    # try Schur complement
    # IS0 = IS().createGeneral(index_0)
    # IS1 = IS().createGeneral(index_1)
    # ksp = KSP().create()
    # ksp.setType(KSP.Type.PREONLY)
    # pc = ksp.getPC()
    # pc.setType(PC.Type.HYPRE)
    # pc.setType(PC.Type.FIELDSPLIT)
    # pc.setFieldSplitType(PC.CompositeType.MULTIPLICATIVE)
    # pc.setFieldSplitIS(("u0", IS0), ("u1", IS1))
    # pc.setFieldSplitSchurFactType(PC.SchurFactType.FULL)
    # pc.setFieldSplitSchurPreType(PC.SchurPreType.SELFP)
    # sub_ksp = pc.getFieldSplitSubKSP()
    # sub_ksp[0].setType(KSP.Type.PREONLY)
    # sub_ksp[0].getPC().setType(PC.Type.GAMG)
    # sub_ksp[1].setType(KSP.Type.PREONLY)
    # sub_ksp[1].getPC().setType(PC.Type.ILU)
    # ksp.setOperators(A_Mat)
    # ksp.setFromOptions()
    #
    # Y_Vec = A_Mat.createVecRight()
    # start_time = default_timer()
    # ksp.solve(L_Vec, Y_Vec)
    # print("HYPRE = {}s".format(default_timer() - start_time))
    # y = Y_Vec.getArray()

    # print("Difference = {:.2e}".format(np.linalg.norm(z-y, ord=np.inf)))
    
