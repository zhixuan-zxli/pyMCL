from os import path
import numpy as np
from matplotlib import pyplot
from fem import *

cp_group = "result/MCL-B0.05-s{:d}"
cp_base_num = 8192
base_dt = 1.0/1024/32
num_hier = 2

@Functional
def xdy_ydx(x: QuadData) -> np.ndarray:
    # x: (2, Ne, Nq)
    # x.grad: (2, 1, Ne, Nq)
    return 0.5 * (x[0] * x.grad[1,0] - x[1] * x.grad[0,0])[np.newaxis]

@Functional
def c_L2(x: QuadData, u: QuadData) -> np.ndarray:
    # u: (?, Ne, Nq)
    return np.sum(u**2, axis=0, keepdims=True) * x.dx

if __name__ == "__main__":

    table_header = [str(k+1) for k in range(num_hier-1)]
    error_table = {
        "y L^inf": [0.0] * (num_hier-1), 
        "y L^2": [0.0] * (num_hier-1)
    }
    print("Showing t = {}\n".format(cp_base_num * base_dt))

    for k in range(num_hier):
        # prepare the mesh
        if k == 0:
            mesh = Mesh()
            mesh.load("mesh/two-phase.msh")
        else:
            mesh = splitRefine(mesh)
        # setMeshMapping(mesh)
        i_mesh = mesh.view(1, tags=(3, )) # interface mesh
        setMeshMapping(i_mesh)

        Y_sp = i_mesh.coord_fe # type: FunctionSpace

        # read the data from checkpoint
        cp_file = path.join(cp_group.format(k), "{:04d}.npz".format(cp_base_num * 2**k))
        cp = np.load(cp_file)
        energy = cp["energy"] # type: np.ndarray
        refcl_hist = cp["refcl_hist"] # type: np.ndarray
        phycl_hist = cp["phycl_hist"] # type: np.ndarray

        # plot the total energy
        t_span = np.arange(energy.shape[0]) * base_dt / 2**k
        # pyplot.figure()
        # pyplot.plot(t_span, np.sum(energy, axis=1), '-', label="total")
        # pyplot.legend()
        # pyplot.title(cp_file)
        # plot the contact line motion
        pyplot.figure()
        pyplot.plot(t_span, refcl_hist[:,0], '-', label="ref left")
        pyplot.plot(t_span, refcl_hist[:,2], '-', label="ref right")
        pyplot.plot(t_span, phycl_hist[:,0], '-', label="phy left")
        pyplot.plot(t_span, phycl_hist[:,2], '-', label="phy right")
        pyplot.legend()
        pyplot.title(cp_file)

        # extract the interface parametrization
        y_k = Function(Y_sp)
        y_k[:] = cp["y_k"]
        y_k_err = Function(Y_sp)
        # calculate the volume
        ds = Measure(i_mesh, dim=1, order=3)
        vol_y = xdy_ydx.assemble(Measure(i_mesh, dim=1, order=3, coord_map=y_k))
        if k > 0:
            # Interpolate y_k_prev onto Y_sp. 
            # According to the implementation of splitRefine, 
            # the refined elements are ordered interlaced. 
            gdim = 2
            for d in range(gdim):
                y_k_err[Y_sp.elem_dof[0*gdim+d, ::2]] = y_k_prev[Y_sp_prev.elem_dof[0*gdim+d]]
                y_k_err[Y_sp.elem_dof[1*gdim+d, 1::2]] = y_k_prev[Y_sp_prev.elem_dof[1*gdim+d]]
                y_k_err[Y_sp.elem_dof[1*gdim+d, ::2]] = 0.5 * (y_k_prev[Y_sp_prev.elem_dof[0*gdim+d]] + y_k_prev[Y_sp_prev.elem_dof[1*gdim+d]])
            y_k_err = y_k - y_k_err # type: Function
            # calculate the error
            error_table["y L^inf"][k-1] = np.linalg.norm(y_k_err, ord=np.inf)
            error_table["y L^2"][k-1] = np.sqrt(c_L2.assemble(ds, u=y_k_err._interpolate(ds))).item()
        y_k_prev = y_k # type: Function
        Y_sp_prev = Y_sp # type: FunctionSpace

    printConvergenceTable(table_header, error_table)

    pyplot.show()
