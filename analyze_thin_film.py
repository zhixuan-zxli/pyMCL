import pickle
import numpy as np
from scipy.linalg import solve as dense_solve
from scipy.special import expi
from fem.post import printConvergenceTable
from thin_film import downsample, PhysicalParameters
from matplotlib import pyplot
import warnings

def getTimeConvergence() -> None:
    num_hier = 4
    filenames = "result/tf-s-2-g4-s1t{}/0032.npz"
    data = []
    table_headers = ["T{}".format(i+2) for i in range(num_hier-1)]
    error_table = { "h inf": [], "g inf": [], "a": [] }
    for i in range(num_hier):
        data.append(np.load(filenames.format(i+2)))
    for i in range(num_hier-1):
        h_diff = data[i+1]["h"] - data[i]["h"]
        error_table["h inf"].append(np.linalg.norm(h_diff[2:-1], ord=np.inf))
        g_diff = data[i+1]["g"] - data[i]["g"]
        error_table["g inf"].append(np.linalg.norm(g_diff[1:-1], ord=np.inf))
        a_diff = data[i+1]["a_hist"][1,-1] - data[i]["a_hist"][1,-1]
        error_table["a"].append(np.abs(a_diff).item())
    print("\nTime convergence: ")
    printConvergenceTable(table_headers, error_table)

def getSpaceConvergence() -> None:
    num_hier = 4
    base_grid = 128
    filenames = "result/tf-s-2-g4-s{}t{}/0032.npz"
    data = []
    table_headers = ["1/{}".format(base_grid*2**i) for i in range(num_hier-1)]
    error_table = {"h": [], "g": [], "a": []}
    for i in range(num_hier):
        data.append(np.load(filenames.format(i, 2*i)))
    for i in range(num_hier-1):
        h_diff = downsample(data[i+1]["h"], 2) - data[i]["h"]
        error_table["h"].append(np.linalg.norm(h_diff[2:-1], ord=np.inf))
        g_diff = downsample(data[i+1]["g"], 1) - data[i]["g"]
        error_table["g"].append(np.linalg.norm(g_diff[2:-1], ord=np.inf))
        a_diff = data[i+1]["a_hist"][1,-1] - data[i]["a_hist"][1,-1]
        error_table["a"].append(np.abs(a_diff).item())
    print("\nSpace convergence: ")
    printConvergenceTable(table_headers, error_table)

def plotCLSpeed() -> None:
    datanames = ["result/tf-s-6-g4-aa/0008.npz"]
    data = [np.load(name) for name in datanames]
    # plot the contact line location
    _, ax1 = pyplot.subplots()
    _, ax2 = pyplot.subplots()
    for name, npz in zip(datanames, data):
        a_hist = npz["a_hist"]
        ax1.plot(a_hist[0], a_hist[1], '-', label=name)
        speed = (a_hist[1,1:] - a_hist[1,:-1]) / (a_hist[0,1:] - a_hist[0,:-1])
        ax2.plot(a_hist[0,1:-1], speed[1:], '-', label=name)
    ax1.set_ylabel("$a(t)$"); ax1.legend()
    ax2.set_ylabel("$\\dot{a}(t)$"); ax2.set_ylim(0.0, 0.5); ax2.legend()

def plotProfiles() -> None:
    filename = "tf-s-4-g4-aa"
    checkpoints = [2, 4] #[1, 2, 4, 8, 16, 32]
    V0 = 4.0 * (1 - (1-np.exp(-4))/4)
    with open("result/" + filename + "/PhysicalParameters", "rb") as f:
        phyp = pickle.load(f)
    phyp.eps = -1.0 / np.log(phyp.slip)
    print("Parameters of the profiles: \nV0 = {:.4f}\n".format(V0), phyp)
    # load the data
    npzdata = []
    for cp in checkpoints:
        name = "result/" + filename + "/{:04}.npz".format(cp)
        npzdata.append(np.load(name))
    #
    _, ax1 = pyplot.subplots() # axis for plotting the profiles
    _, ax2 = pyplot.subplots() # axis for plotting the slope dh
    _, ax3 = pyplot.subplots() # axis for plotting the slope dg
    alpha_list = np.linspace(1.0, 0.2, len(checkpoints))[::-1]
    for data, alpha in zip(npzdata, alpha_list):
        xi_c = data["xi_c"]
        a_hist = data["a_hist"]
        h = data["h"]
        g = data["g"]
        t = a_hist[0, -1]
        a = a_hist[1, -1]
        adot = (a_hist[1, -1] - a_hist[1, -2]) / (a_hist[0, -1] - a_hist[0, -2])
        print("adot = {:.3e}".format(adot))
        n_fluid = h.size - 3
        ax1.plot(a * xi_c[2:-2], g[1:-1], "-", color="tab:blue", label="$t={:.2f}$".format(t), alpha=alpha)
        ax1.plot(a * xi_c[2:2+n_fluid], h[2:-1], "-", color="tab:blue", alpha=alpha)
        # calculate and plot the slopes
        z = np.log(a * (1.0 - xi_c[2:n_fluid+2])) * phyp.eps + 1.0
        dh = (h[3:n_fluid+3] - h[1:n_fluid+1]) / (xi_c[3:n_fluid+3] - xi_c[1:n_fluid+1]) / a
        dg = (g[2:n_fluid+2] - g[:n_fluid]) / (xi_c[3:n_fluid+3] - xi_c[1:n_fluid+1]) / a
        ax2.plot(z, -dh, 'o', markerfacecolor='none', label="$t={:.2f}$".format(t), alpha=alpha)
        ax3.plot(z, -dg, 'o', markerfacecolor='none', label="$t={:.2f}$".format(t), alpha=alpha)
        # plot the theoretical prediction
        plotPrediction(xi_c[2:2+n_fluid], xi_c[2:-2], phyp, V0, a, adot, alpha, ax1, ax2, ax3)
    ax1.legend()
    ax2.legend(); ax2.set_xlabel("$z$"); ax2.set_ylabel("$h'$")
    ax3.legend(); ax3.set_xlabel("$z$"); ax3.set_ylabel("$g'$")
    # pyplot.show()
    # fig.savefig("thin films.png", dpi=300)

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

def plotPrediction(xi_f: np.ndarray, xi_s: np.ndarray, phyp: PhysicalParameters, V0: float, a: float, adot: float, alpha: float, ax1, ax2, ax3) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eps, th_y = phyp.eps, phyp.theta_Y
        # estimate from asymptotic relation
        a_app = 3*V0 / (a**2 * (1+1/phyp.gamma[0]))
        b_app = -1/phyp.gamma[0] * a_app
        th_app = 3*V0 / a**2
        B0 = phyp.bm / eps**2 #np.sqrt(phyp.bm / phyp.gamma[0]) / eps
        b_til = np.sqrt(phyp.gamma[0]) / (np.sqrt(phyp.gamma[0]) + np.sqrt(phyp.gamma[1])) * b_app
        a0 = ((a_app - b_til)**3 - th_y**3) / 3
        ea1 = adot / eps - a0
        # ================== Outer region ==================
        h0 = a_app * a * (1.0 - xi_f**2) / 2
        xf = a * xi_f
        # h1 = eps * a0 / th_app**2 * ((a+xf) * np.log(a+xf) + (a-xf)*np.log(a-xf) - 2*a*np.log(2*a) + 3*a/2*(1.0-xi_f**2)) # the first order correction
        xs = a * xi_s
        g0 = np.where(xi_s <= 1.0, b_app * a * (1.0 - xi_s**2) / 2, 0.0)
        # g1 = np.where(xi_s <= 1.0, \
        #     -eps * a0 / phyp.gamma[0] / th_app**2 * ((a+xs) * np.log(a+xs) + (a-xs)*np.log(a-xs) - 2*a*np.log(2*a) + 3*a/2*(1.0-xi_s**2)), \
        #     0.0)
        # ax1.plot(xf, h0 + h1, '--', color='k', alpha=alpha)
        # ax1.plot(xs, g0 + g1, '--', color='k', alpha=alpha)
        y = a - xf
        z = np.log(y) * eps + 1.0
        ax2.plot(z, a_app * xf / a + eps * a0 / th_app**2 * (np.log(y) + (3-np.log(2*a))), 'c-', alpha=alpha)
        # ax3.plot(z, b_app * xf / a - eps * a0 / phyp.gamma[0] / th_app**2 * (np.log(y) + (3-np.log(2*a))), 'c:', alpha=alpha)

        # ================== Inner region ==================
        s = y / phyp.slip
        ax2.plot(z, th_y + b_til + eps * a0 / th_y * (np.log(th_y*s + 1) / th_y + s * np.log(1 + 1/(th_y*s)) + 1), 'g-', alpha=alpha)

        # ================== Intermediate region ==================
        m0 = (th_y**3 + 3*a0*z)**(1/3)
        m01 = m0 + 1 / m0**2 * (ea1*z + eps * a0 * (th_y + np.log(th_y) + 1))
        flag = z > 0.0
        ax2.plot(z[flag], m01[flag] + b_til, 'm-', alpha=alpha)
        
        # ================== Bending region ==================
        gamma = phyp.gamma
        # calculate the constants in the leading-order solution
        lb = [np.sqrt(B0 / gam) for gam in gamma] 
        C_neg = lb[1] * np.sqrt(gamma[0]) / (np.sqrt(gamma[0]) + np.sqrt(gamma[1])) * b_app
        C_pos = lb[0] * np.sqrt(gamma[1]) / (np.sqrt(gamma[0]) + np.sqrt(gamma[1])) * b_app
        D = C_neg - C_pos
        y_til = (a - xs) / eps
        g_til_0 = np.where(y_til >= 0.0, 
                        D + b_app * y_til + C_pos * np.exp(-y_til / lb[0]), 
                        C_neg * np.exp(y_til / lb[1]))
        # calculate the constants in the next-order solution
        C1_til, C0_til = -a_app / a, 3 + np.log(eps) - np.log(2*a)
        A = np.array(((gamma[0], -gamma[1]), (1.0/lb[0], 1.0/lb[1])))
        b = np.array((B0/gamma[0]*C1_til, adot / (gamma[0] * th_app**2) * (0.57721 - np.log(lb[0]) - C0_til)))
        D = dense_solve(A, b)
        E1 = adot / th_app**2 * (-np.log(lb[0]) - C0_til)
        F1 = gamma[0] / lb[0] * (D[1] - D[0])
        z1 = np.maximum(0.0, y_til / lb[0])
        g_til_1 = np.where(y_til >= 0.0, 
                        D[0]*np.exp(-z1) + lb[0] / gamma[0] * (-C1_til*lb[0]/2*z1**2 + E1*z1 + F1 + adot/th_app**2*(phi(z1) - z1*(np.log(z1)-1))), 
                        D[1]*np.exp(y_til / lb[1]))
        ax1.plot(xs, (g_til_0 + eps * g_til_1) * eps, 'k:', alpha=alpha)
        z1 = z1[z1 > 0.0]
        dg_til_0 = b_app - C_pos / lb[0] * np.exp(-z1)
        dg_til_1 = (-D[0]*np.exp(-z1) + lb[0]/gamma[0]*(adot/th_app**2*(dphi(z1) - np.log(z1)) - C1_til*lb[0]*z1 + E1)) / lb[0]
        dh_til_1 = C1_til * (z1*lb[0]) + a0/th_app**2*(np.log(z1*lb[0]) + (3.0+np.log(eps)-np.log(2*a)))
        ax3.plot(np.log(z1 * lb[0] * eps) * eps + 1.0, dg_til_0 + eps * dg_til_1, 'r-', alpha=alpha)
        ax2.plot(np.log(z1 * lb[0] * eps) * eps + 1.0, a_app + eps * (dh_til_1), 'r-', alpha=alpha) # y_til -> +infty    

if __name__ == "__main__":

    # getTimeConvergence()
    # getSpaceConvergence()
    # plotCLSpeed()
    plotProfiles()

    pyplot.show()
