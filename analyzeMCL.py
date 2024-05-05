from os import path
import numpy as np
from matplotlib import pyplot
from fem import *

cp_group = "result/MCL-B0.05-s{:d}"
cp_base_num = 4096
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

def lift_to_P2(P2_space: FunctionSpace, p1_func: Function) -> Function:
    p2_func = Function(P2_space)
    p2_func[:p1_func.size] = p1_func
    rdim = P2_space.elem.rdim
    assert rdim == p1_func.fe.elem.rdim
    if P2_space.mesh.tdim == 1:
        for d in range(rdim):
            p2_func[P2_space.elem_dof[2*rdim+d]] = 0.5 * (p1_func[P2_space.elem_dof[d]] + p1_func[P2_space.elem_dof[rdim+d]])
    else:
        raise NotImplementedError
    return p2_func

def interp_P1(fine_fs: FunctionSpace, coarse_fs: FunctionSpace, y: Function) -> Function:
    # y: Function on the coarse space
    rdim = fine_fs.elem.rdim
    y_interp = Function(fine_fs)
    fine_dof = fine_fs.elem_dof
    coarse_dof = coarse_fs.elem_dof    
    # According to the implementation of splitRefine, 
    # the refined elements are ordered interlaced. 
    for d in range(rdim):
        y_interp[fine_dof[0*rdim+d, ::2]] = y[coarse_dof[0*rdim+d]]
        y_interp[fine_dof[1*rdim+d, 1::2]] = y[coarse_dof[1*rdim+d]]
        y_interp[fine_dof[1*rdim+d, ::2]] = 0.5 * (y[coarse_dof[0*rdim+d]] + y[coarse_dof[1*rdim+d]])
    return y_interp

if __name__ == "__main__":

    table_header = [str(k+1) for k in range(num_hier-1)]
    error_table = {
        "y L^inf": [0.0] * (num_hier-1), 
        "y L^2": [0.0] * (num_hier-1), 
        "vol": [0.0] * (num_hier-1),
    }
    print("\n * Showing t = {}".format(cp_base_num * base_dt))

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
        s_mesh = mesh.view(1, tags=(4, 5)) # sheet reference mesh
        setMeshMapping(s_mesh)

        Y_sp = i_mesh.coord_fe # type: FunctionSpace
        Q_P1_sp = s_mesh.coord_fe # type: FunctionSpace
        Q_sp = FunctionSpace(s_mesh, VectorElement(LineP2, 2)) # for deformation and also for the fluid stress

        # read the data from checkpoint
        cp_file = path.join(cp_group.format(k), "{:04d}.npz".format(cp_base_num * 2**k))
        cp = np.load(cp_file)
        energy = cp["energy"] # type: np.ndarray
        refcl_hist = cp["refcl_hist"] # type: np.ndarray
        phycl_hist = cp["phycl_hist"] # type: np.ndarray

        # plot the total energy
        if k == num_hier-1:
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

        # extract the interface parametrization
        y_k = Function(Y_sp)
        y_k[:] = cp["y_k"]

        # calculate the volume
        vol = xdy_ydx.assemble(Measure(i_mesh, dim=1, order=3, coord_map=y_k))
        q_k = Function(Q_sp)
        q_k[:] = cp["w_k"]
        id_k = Function(Q_P1_sp)
        id_k[:] = cp["id_k"]
        q_k += lift_to_P2(Q_sp, id_k)
        vol += xdy_ydx.assemble(Measure(s_mesh, dim=1, order=5, coord_map=q_k)) # type: float

        if k > 0:
            # calculate the error of the interface
            # Interpolate y_k_prev onto Y_sp:
            y_k_interp = interp_P1(Y_sp, Y_sp_prev, y_k_prev)
            y_k_err = y_k - y_k_interp # type: Function
            ds = Measure(i_mesh, dim=1, order=3)
            error_table["y L^2"][k-1] = np.sqrt(c_L2.assemble(ds, u=y_k_err._interpolate(ds)))
            error_table["y L^inf"][k-1] = np.linalg.norm(y_k_err, ord=np.inf)
            # calculate the error of the volume
            error_table["vol"][k-1] = np.abs(vol - vol_prev)
        y_k_prev = y_k # type: Function
        Y_sp_prev = Y_sp # type: FunctionSpace
        vol_prev = vol

    printConvergenceTable(table_header, error_table)
    pyplot.show()
