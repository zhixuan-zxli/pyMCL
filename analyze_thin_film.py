import pickle
import numpy as np
from os.path import join as pjoin
from scipy.linalg import solve as dense_solve
from scipy.special import expi
from fem.post import printConvergenceTable
from thin_film import PhysicalParameters
from matplotlib import pyplot
import warnings

def getTimeConvergence() -> None:
    num_hier = 4
    filenames = "result/tf-s-2-g2-uni-s1t{}/0004.npz"
    data = []
    table_headers = ["T{}".format(i+1) for i in range(num_hier-1)]
    error_table = { "h inf": [], "g inf": [], "kappa inf": [], "a": [] }
    for i in range(num_hier):
        data.append(np.load(filenames.format(i+1)))
    for i in range(num_hier-1):
        h_diff = data[i+1]["h"] - data[i]["h"]
        error_table["h inf"].append(np.linalg.norm(h_diff[2:-1], ord=np.inf))
        g_diff = data[i+1]["g"] - data[i]["g"]
        error_table["g inf"].append(np.linalg.norm(g_diff[1:-1], ord=np.inf))
        k_diff = data[i+1]["kappa"] - data[i]["kappa"]
        error_table["kappa inf"].append(np.linalg.norm(k_diff[1:-1], ord=np.inf))
        a_diff = data[i+1]["a_hist"][1,-1] - data[i]["a_hist"][1,-1]
        error_table["a"].append(np.abs(a_diff).item())
    print("\nTime convergence: ")
    printConvergenceTable(table_headers, error_table)

def downsample(u: np.ndarray, ng_left: int) -> np.ndarray:
    """
    ng_left [int] number of ghosts on the left
    """
    usize = u.size
    u_down = np.zeros(((usize-ng_left-1)//2 + ng_left+1, ))
    u_down[ng_left:-1] = (u[ng_left:-2:2] + u[ng_left+1:-1:2]) / 2
    # symmetry condition at the left
    for i in range(ng_left):
        u_down[i] = u_down[2*ng_left-i-1]
    return u_down

def getSpaceConvergence() -> None:
    num_hier = 4
    base_grid = 128
    filenames = "result/tf-s-2-g2-uni-s{}t{}/0004.npz"
    data = []
    table_headers = ["1/{}".format(base_grid*2**i) for i in range(num_hier-1)]
    error_table = {"h": [], "g": [], "kappa": [], "a": []}
    for i in range(num_hier):
        data.append(np.load(filenames.format(i, i)))
    for i in range(num_hier-1):
        h_diff = downsample(data[i+1]["h"], 2) - data[i]["h"]
        error_table["h"].append(np.linalg.norm(h_diff[2:-1], ord=np.inf))
        g_diff = downsample(data[i+1]["g"], 1) - data[i]["g"]
        error_table["g"].append(np.linalg.norm(g_diff[2:-1], ord=np.inf))
        k_diff = downsample(data[i+1]["kappa"], 1) - data[i]["kappa"]
        error_table["kappa"].append(np.linalg.norm(k_diff[2:-1], ord=np.inf))
        a_diff = data[i+1]["a_hist"][1,-1] - data[i]["a_hist"][1,-1]
        error_table["a"].append(np.abs(a_diff).item())
    print("\nSpace convergence: ")
    printConvergenceTable(table_headers, error_table)

def plotCLSpeed() -> None:
    cp_list = ["result/tf-s-4-g8-aa/0008.npz", "result/tf-s-4-g4-aa/0008.npz", 
               "result/tf-s-4-g2-aa/0008.npz", "result/tf-s-4-g1-aa/0008.npz"]
    labels = []; npzdata = []; params = []
    # load also the parameters
    for cp in cp_list:
        parts = cp.split("/")
        labels.append(parts[-2])
        npzdata.append(np.load(cp))
        with open(pjoin(*parts[:-1], "PhysicalParameters"), "rb") as f:
            params.append(pickle.load(f))
            params[-1].eps = -1.0 / np.log(params[-1].slip)
    _, ax1 = pyplot.subplots()
    _, ax2 = pyplot.subplots()
    for label, npz, phyp in zip(labels, npzdata, params):
        print("Plotting", phyp)
        a_hist = npz["a_hist"]
        ax1.plot(a_hist[0], a_hist[1], '-', label=label)
        speed = (a_hist[1,1:] - a_hist[1,:-1]) / (a_hist[0,1:] - a_hist[0,:-1])
        a = a_hist[1,1:]
        ax2.plot(a[1::4096], speed[1::4096], 'o', label=label, alpha=0.4)
        # calculate and plot the theoretical prediction
        a_app = 3*phyp.vol/(a**2*(1+1/phyp.gamma[0]))
        b_app = -a_app / phyp.gamma[0]
        b_til = np.sqrt(phyp.gamma[0]) / (np.sqrt(phyp.gamma[0]) + np.sqrt(phyp.gamma[1])) * b_app
        a_est = phyp.eps * ((a_app - b_til)**3 - phyp.theta_Y**3) / 3
        ax2.plot(a[1:], a_est[1:], '-')
    ax1.set_xlabel("$t$"); ax1.set_ylabel("$a(t)$"); ax1.legend()
    ax2.set_xlabel("$a$"); ax2.set_ylabel("$\\dot{a}(t)$"); ax2.legend()

def plotProfiles() -> None:
    filename = "tf-s-4-g2-Y1-rec"
    cp_list = [8]
    with open("result/" + filename + "/PhysicalParameters", "rb") as f:
        phyp = pickle.load(f)
    phyp.eps = -1.0 / np.log(phyp.slip)
    print("Parameters of the profiles:", phyp)
    # load the data
    npzdata = []
    for cp in cp_list:
        name = "result/" + filename + "/{:04}.npz".format(cp)
        npzdata.append(np.load(name))
    #
    _, ax1 = pyplot.subplots() # axis for plotting the profiles
    _, ax2 = pyplot.subplots() # axis for plotting the slope dh
    _, ax3 = pyplot.subplots() # axis for plotting the slope dg
    alpha_list = np.linspace(1.0, 0.2, len(cp_list))[::-1]
    for cp, data, alpha in zip(cp_list, npzdata, alpha_list):
        xi_c = data["xi_c"]
        a_hist = data["a_hist"]
        h = data["h"]
        g = data["g"]
        t = a_hist[0, -1]
        a = a_hist[1, -1]
        n_fluid = h.size - 3
        adot = (a_hist[1,1:] - a_hist[1,:-1]) / (a_hist[0,1:] - a_hist[0,:-1])
        print("Plotting checkpoint {}, adot = {:.3e}".format(cp, adot[-1]))
        label = "$t={:.1f}$".format(t)
        # plot the profiles
        ax1.plot(a * xi_c[2:-2], g[1:-1], '-', color="tab:orange", label=label, alpha=alpha)
        ax1.plot(a * xi_c[2:2+n_fluid], h[2:-1], '-', color="tab:blue", alpha=alpha)
        # ax1.plot(a * xi_c[2:-2:4], g[1:-1:4], "o", mfc='none', mec="tab:orange", label=label, alpha=alpha)
        # ax1.plot(a * xi_c[2:2+n_fluid:4], h[2:-1:4], "o", mfc='none', mec="tab:blue", alpha=alpha)
        # calculate and plot the slopes
        z = np.log(a * (1.0 - xi_c[2:n_fluid+2])) * phyp.eps + 1.0
        dh = (h[3:n_fluid+3] - h[1:n_fluid+1]) / (xi_c[3:n_fluid+3] - xi_c[1:n_fluid+1]) / a
        dg = (g[2:n_fluid+2] - g[:n_fluid]) / (xi_c[3:n_fluid+3] - xi_c[1:n_fluid+1]) / a
        ax2.plot(z[::2], -dh[::2], 'o', mfc='none', mec="tab:blue", label=label, alpha=alpha)
        ax3.plot(z[::2], -dg[::2], 'o', mfc='none', mec="tab:orange", label=label, alpha=alpha)
        # plot the theoretical prediction
        plotPrediction(xi_c[2:2+n_fluid], xi_c[2:-2], phyp, a, adot[-1], alpha, ax1, ax2, ax3, cp == cp_list[-1])
    ax1.legend(); ax1.set_xlabel("$x$"); ax1.set_xlim(0.0, 2.5); ax1.set_ylabel("$z$")
    ax2.legend(); ax2.set_xlabel("$z$"); ax2.set_ylabel("$h'$")
    ax3.legend(); ax3.set_xlabel("$z$"); ax3.set_ylabel("$g'$")
    # fig.savefig("thin films.png", dpi=300)

# these are the specific solutions appeared in the next-order solution of the bending problem
def phi(x: np.ndarray) -> np.ndarray:
    r1 = np.exp(x) * expi(-x)
    r2 = np.exp(-x) * expi(x)
    r = (r1 - r2) / 2
    r[x == 0.0] = 0.0
    return r

def dphi(x: np.ndarray) -> np.ndarray:
    r1 = np.exp(x) * expi(-x)
    r2 = np.exp(-x) * expi(x)
    r = (r1 + r2) / 2
    r[x == 0.0] = 0.0
    return r

def plotPrediction(xi_f: np.ndarray, xi_s: np.ndarray, phyp: PhysicalParameters, a: float, adot: float, alpha: float, ax1, ax2, ax3, lab: bool) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eps, th_y = phyp.eps, phyp.theta_Y
        # estimate from asymptotic relation
        a_app = 3*phyp.vol / (a**2 * (1+1/phyp.gamma[0]))
        b_app = -1/phyp.gamma[0] * a_app
        th_app = 3*phyp.vol / a**2
        Cb = phyp.bm / eps**2 #np.sqrt(phyp.bm / phyp.gamma[0]) / eps
        b_til = np.sqrt(phyp.gamma[0]) / (np.sqrt(phyp.gamma[0]) + np.sqrt(phyp.gamma[1])) * b_app
        a0 = ((a_app - b_til)**3 - th_y**3) / 3
        ea1 = adot / eps - a0
        # ================== Outer region ==================
        h0 = a_app * a * (1.0 - xi_f**2) / 2
        xf = a * xi_f
        h1 = a0 / th_app**2 * ((a+xf) * np.log(a+xf) + (a-xf)*np.log(a-xf) - 2*a*np.log(2*a) + 3*a/2*(1.0-xi_f**2)) # the first order correction
        xs = a * xi_s
        g0 = np.where(xi_s <= 1.0, b_app * a * (1.0 - xi_s**2) / 2, 0.0)
        g1 = np.where(xi_s <= 1.0, \
            -a0 / phyp.gamma[0] / th_app**2 * ((a+xs) * np.log(a+xs) + (a-xs)*np.log(a-xs) - 2*a*np.log(2*a) + 3*a/2*(1.0-xi_s**2)), \
            0.0)
        # ax1.plot(xf, h0 + eps * h1, 'k-', label="Asymptotics" if lab else None)
        # ax1.plot(xs, g0 + eps * g1, 'k-')
        y = a - xf
        z = np.log(y) * eps + 1.0
        dh_outer = a_app * xf / a + eps * a0 / th_app**2 * (np.log(y) + (3-np.log(2*a)))
        mask = z > 0.7
        ax2.plot(z[mask], dh_outer[mask], 'k-', label="Outer" if lab else None)

        # ================== Inner region ==================
        s = y / phyp.slip
        dh_inner = th_y + b_til + eps * a0 / th_y * (np.log(th_y*s + 1) / th_y + s * np.log(1 + 1/(th_y*s)) + 1)
        mask = z < 0.2
        ax2.plot(z[mask], dh_inner[mask], 'm--', label="Inner" if lab else None)

        # ================== Intermediate region ==================
        m0 = (th_y**3 + 3*a0*z)**(1/3)
        m01 = m0 + 1 / m0**2 * (ea1*z + eps * a0 * (th_y + np.log(th_y) + 1))
        mask = (z > 0.0) & (z < 0.5)
        ax2.plot(z[mask], m01[mask] + b_til, 'm-', label="Intermediate" if lab else None)
        
        # ================== Bending region ==================
        gamma = phyp.gamma
        # calculate the constants in the leading-order solution
        lb = [np.sqrt(Cb / gam) for gam in gamma] 
        C2 = lb[1] * np.sqrt(gamma[0]) / (np.sqrt(gamma[0]) + np.sqrt(gamma[1])) * b_app
        C1 = lb[0] * np.sqrt(gamma[1]) / (np.sqrt(gamma[0]) + np.sqrt(gamma[1])) * b_app
        A = C2 - C1
        y_til = (a - xs) / eps
        g_til_0 = np.where(y_til >= 0.0, 
                        A + b_app * y_til + C1 * np.exp(-y_til / lb[0]), 
                        C2 * np.exp(y_til / lb[1]))
        # calculate the constants in the next-order solution
        E1 = np.log(2*a/(eps*lb[0])) - 3
        A = np.array(((1.0, -1.0), (lb[0], lb[1])))
        rhs = np.array((b_app / a, a0/(gamma[0]*th_app**2)*(0.57722 + E1)))
        D = dense_solve(A, rhs)
        F1 = Cb * (D[1]/gamma[1] - D[0]/gamma[0])
        y1 = np.maximum(0.0, y_til / lb[0])
        g_til_1 = np.where(y_til >= 0.0, 
                        D[0]*lb[0]**2*np.exp(-y1) - b_app/(2*a)*y_til**2 + a0*lb[0]/(gamma[0]*th_app**2)*(phi(y1) - y1*np.log(y1) + (E1+1)*y1) + F1, 
                        D[1]*lb[1]**2*np.exp(y_til / lb[1]))
        #
        h_til_0 = a_app * y_til + C2
        h_til_1 = a0/th_app**2 * (y_til * np.log(y_til) + (2 + np.log(eps/(2*a)))*y_til) - a_app/(2*a)*y_til**2 + D[1]*lb[1]
        # ax1.plot(xs, (h_til_0 + eps * h_til_1) * eps, 'k-', label="Asymptotics" if lab else None)
        # ax1.plot(xs, (g_til_0 + eps * g_til_1) * eps, 'k-')
        # plot the slopes
        y_til = y_til[y_til > 0.0]
        y1 = y1[y1 > 0.0]
        z = np.log(y_til * eps) * eps + 1.0
        mask = (z > 0.3) & (z < 0.7)
        dg_til_0 = b_app - C1 / lb[0] * np.exp(-y1)
        dg_til_1 = -D[0]*lb[0]*np.exp(-y1) - b_app/a*y_til + a0/(gamma[0]*th_app**2)*(dphi(y1) - np.log(y1) + E1)
        ax3.plot(z, dg_til_0 + eps * dg_til_1, 'k-', label="Asymptotics" if lab else None)
        dh_til_0 = a_app
        dh_til_1 = a0/th_app**2*(np.log(y_til) + 3 + np.log(eps/(2*a))) - a_app/a*y_til
        ax2.plot(z[mask], (dh_til_0 + eps*dh_til_1)[mask], 'k--', label="Bending" if lab else None) # y_til -> +infty    

if __name__ == "__main__":

    # getTimeConvergence()
    # getSpaceConvergence()
    # plotCLSpeed()
    plotProfiles()

    pyplot.show()
