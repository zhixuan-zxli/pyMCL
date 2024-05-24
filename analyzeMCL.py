import numpy as np
from matplotlib import pyplot
from fem import *

mesh_name = "mesh/two-phase-a120.msh"
cp_group = "result/MCL-adv-s{}t{}/{:04d}.npz"
base_step = 256
base_dt = 1.0/256
ref_level = ((2,2), (2,3), (2,4), (2,5)) # (spatial, time) for each pair
num_hier = len(ref_level)

@Functional
def xdy_ydx(x: QuadData) -> np.ndarray:
    # x: (2, Ne, Nq)
    # x.grad: (2, 1, Ne, Nq)
    return 0.5 * (x[0] * x.grad[1,0] - x[1] * x.grad[0,0])[np.newaxis]

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

def down_to_P1(P1_space: FunctionSpace, p2_func: Function) -> Function:
    p1_func = Function(P1_space)
    p1_func[:] = p2_func[:P1_space.num_dof]
    return p1_func

def error_between_interface(y_coarse: Function, y_fine: Function) -> float:
    elem_dof = y_fine.fe.elem_dof
    segs = y_fine[elem_dof].view(np.ndarray) # (4, Ne)
    segs = segs.T.reshape(-1, 2, 2) # (Ne, 2, 2)
    segs_dir = segs[:,1] - segs[:,0]
    segs_norm = np.linalg.norm(segs_dir, axis=1) # (Ne, )
    segs_dir = segs_dir / segs_norm[:, np.newaxis]
    y_c = y_coarse.reshape(-1, 2)

    def point2polyline(p: np.ndarray) -> float:
        a = segs - p[np.newaxis, np.newaxis] # (Ne, 2, 2)
        a = np.linalg.norm(a.reshape(-1,2), axis=1) # (Ne*2, )
        b = p[np.newaxis] - segs[:,0] # (Ne, 2)
        b = np.sum(b * segs_dir, axis=1)
        b = np.where(b < 0.0, np.inf, b)
        b = np.where(b > segs_norm, np.inf, b)
        b = segs[:,0] + b[:, np.newaxis] * segs_dir
        b = np.linalg.norm(b - p[np.newaxis], axis=1) # (Ne, )
        return np.min(np.concatenate((a, b)))
    
    e = [point2polyline(p.view(np.ndarray)) for p in y_c]
    return max(e)


if __name__ == "__main__":

    table_header = [str(k) for k in range(num_hier-1)]
    error_table = {
        "y": [0.0] * (num_hier-1), 
        "q": [0.0] * (num_hier-1),
        "vol": [0.0] * (num_hier-1),
    }
    # print("\n * Showing t = {}".format(cp_base_num * base_dt))
    marker_styles = ("bo", "m+", "rx", "y*")
    ax = pyplot.subplot()
    ax.axis("equal")

    for k in range(num_hier):
        # prepare the mesh
        if k == 0:
            mesh = Mesh()
            mesh.load(mesh_name)
            for _ in range(ref_level[0][0]):
                mesh = splitRefine(mesh)
        else:
            for _ in range(ref_level[k][0] - ref_level[k-1][0]):
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
        cp_file = cp_group.format(ref_level[k][0], ref_level[k][1], base_step * 2**ref_level[k][1])
        cp = np.load(cp_file)
        energy = cp["energy"] # type: np.ndarray
        refcl_hist = cp["refcl_hist"] # type: np.ndarray
        phycl_hist = cp["phycl_hist"] # type: np.ndarray

        # plot the total energy
        if k == num_hier-1:
            t_span = np.arange(energy.shape[0]) * base_dt / 2**ref_level[k][1]
            # pyplot.figure()
            # pyplot.plot(t_span, np.sum(energy, axis=1), '-', label="total")
            # pyplot.legend()
            # pyplot.title(cp_file)
            # # plot the contact line motion
            # pyplot.figure()
            # pyplot.plot(t_span, refcl_hist[:,0], '-', label="ref left")
            # pyplot.plot(t_span, refcl_hist[:,2], '-', label="ref right")
            # pyplot.plot(t_span, phycl_hist[:,0], '-', label="phy left")
            # pyplot.plot(t_span, phycl_hist[:,2], '-', label="phy right")
            # pyplot.legend()
            pass

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
        vol += xdy_ydx.assemble(Measure(s_mesh, dim=1, order=5, tags=(5,), coord_map=q_k)) # type: float
        q_k_down = down_to_P1(Q_P1_sp, q_k)
        # print("volume at level {} = {}".format(k, vol))

        ax.plot(y_k[::2], y_k[1::2], marker_styles[k], label=str(k))
        ax.plot(q_k[::2], q_k[1::2], marker_styles[k], label=str(k))

        if k > 0:
            error_table["y"][k-1] = error_between_interface(y_k_prev, y_k)
            error_table["q"][k-1] = error_between_interface(q_k_prev, q_k_down)
            error_table["vol"][k-1] = np.abs(vol - vol_prev)
            # error_table["id"][k-1] = np.linalg.norm(id_k - id_k_prev, ord=np.inf)
        # keep the results for the next level
        y_k_prev = y_k.view(np.ndarray).copy() # type: np.ndarray
        q_k_prev = q_k_down.view(np.ndarray).copy() # type: np.ndarray
        vol_prev = vol

    print(f"base_step = {base_step}")
    if num_hier >= 2:
        printConvergenceTable(table_header, error_table)
    ax.legend()
    pyplot.show()
