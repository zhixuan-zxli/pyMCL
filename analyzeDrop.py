import numpy as np
from matplotlib import pyplot
from fem import *
from testDrop import arrange_as_FD

mesh_name = "mesh/half_drop-sq.msh"
cp_group = "result/drop-Y90-flat-s{}t{}/{:05d}.npz"
base_step = 256
base_dt = 1.0/1024
ref_level = ((0,0), (1,2), (2,4), (3,6)) # (spatial, time) for each pair
num_hier = len(ref_level)

@Functional
def xdy_ydx(x: QuadData) -> np.ndarray:
    # x: (2, Ne, Nq)
    # x.grad: (2, 1, Ne, Nq)
    return 0.5 * (x[0] * x.grad[1,0] - x[1] * x.grad[0,0])[np.newaxis]

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
        b = np.minimum(np.maximum(b, 0.0), segs_norm)
        b = segs[:,0] + b[:, np.newaxis] * segs_dir
        b = np.linalg.norm(b - p[np.newaxis], axis=1) # (Ne, )
        return np.min(np.concatenate((a, b)))
    
    e = [point2polyline(p.view(np.ndarray)) for p in y_c]
    return max(e)


if __name__ == "__main__":

    table_header = [str(k) for k in range(num_hier-1)]
    error_table = {
        "r": [0.0] * (num_hier-1), 
        "q": [0.0] * (num_hier-1),
        "vol": [0.0] * (num_hier-1),
    }
    # print("\n * Showing t = {}".format(cp_base_num * base_dt))
    marker_styles = ("bo", "m+", "rx", "y*")
    fig, ax_prof = pyplot.subplots()
    ax_prof.axis("equal")

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

        R_sp = i_mesh.coord_fe # type: FunctionSpace
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
            fig, ax = pyplot.subplots()
            ax.plot(t_span, np.sum(energy, axis=1), '-', label="total")
            ax.legend()
            ax.set_title(cp_file)
            # plot the contact line motion
            fig, ax = pyplot.subplots()
            ax.plot(t_span, refcl_hist[:,0], '-', label="ref cl")
            ax.plot(t_span, phycl_hist[:,0], '-', label="phy cl")
            ax.legend()
            ax.set_title(cp_file)
            # pass

        # extract the interface parametrization
        r_m = Function(R_sp)
        r_m[:] = cp["r_m"]

        # calculate the volume
        vol = xdy_ydx.assemble(Measure(i_mesh, dim=1, order=3, coord_map=r_m))
        q_m = Function(Q_sp)
        q_m[:] = cp["q_m"]
        vol += xdy_ydx.assemble(Measure(s_mesh, dim=1, order=5, tags=(5,), coord_map=q_m)) # type: float
        # print("volume at level {} = {}".format(k, vol))
        q_fd = arrange_as_FD(Q_sp, q_m) # (n, 2)

        ax_prof.plot(r_m[::2], r_m[1::2], marker_styles[k], label=str(k))
        ax_prof.plot(q_m[::2], q_m[1::2], marker_styles[k], label=str(k))

        if k > 0:
            error_table["r"][k-1] = error_between_interface(r_m_prev, r_m)
            if q_fd.shape[0] == q_fd_prev.shape[0]:
                q_err = q_fd - q_fd_prev
            else:
                q_err = q_fd[::2] - q_fd_prev
            error_table["q"][k-1] = np.linalg.norm(q_err, axis=1, ord=None).max()
            error_table["vol"][k-1] = np.abs(vol - vol_prev)
            # error_table["id"][k-1] = np.linalg.norm(id_k - id_k_prev, ord=np.inf)
        # keep the results for the next level
        r_m_prev = r_m.view(np.ndarray).copy() # type: np.ndarray
        q_fd_prev = q_fd.view(np.ndarray).copy() # type: np.ndarray
        vol_prev = vol

    print(f"base_step = {base_step}")
    if num_hier >= 2:
        printConvergenceTable(table_header, error_table)
    ax_prof.legend()
    pyplot.show()
