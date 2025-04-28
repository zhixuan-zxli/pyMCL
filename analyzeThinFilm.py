import pickle
import numpy as np
from os.path import join as pjoin
from scipy.linalg import solve as dense_solve
from scipy.special import expi
from scipy.integrate import solve_ivp
from fem.post import printConvergenceTable
from testThinFilm import PhysicalParameters
from matplotlib import pyplot
from itertools import product
from dataclasses import dataclass
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

def solveLeadOrderODE(phyp: PhysicalParameters, t_span: tuple[float]):
    gamma = np.array(phyp.gamma)
    sqr_gam = np.sqrt(gamma)
    def rhs(t, a):
        alpha = 3*phyp.vol / (a**2 * (1+1/gamma[0]))
        beta = -alpha / gamma[0]
        adot = phyp.eps * ((alpha - sqr_gam[0]/(sqr_gam[0] + sqr_gam[1])*beta)**3 - phyp.theta_Y**3)/3
        return adot
    t_eval = np.linspace(t_span[0], t_span[1], 513)
    solution = solve_ivp(rhs, t_span, [phyp.a_init], t_eval=t_eval, vectorized=True)
    return solution

def plotCLSpeed(cp_list, markers, colors):
    npzdata = []; params = []; 
    # load the checkpoints and the parameters
    for cp in cp_list:
        parts = cp.split("/")
        npzdata.append(np.load(cp))
        with open(pjoin(*parts[:-1], "PhysicalParameters"), "rb") as f:
            params.append(pickle.load(f))
            params[-1].eps = -1.0 / np.log(params[-1].slip)
    _, ax1 = pyplot.subplots()
    _, ax2 = pyplot.subplots()
    for npz, phyp, marker, col in zip(npzdata, params, markers, colors):
        print("Plotting", phyp)
        label = "$\\gamma_1 = {:.1f}$".format(phyp.gamma[0]) # print gamma
        a_hist = npz["a_hist"]
        ax1.plot(a_hist[0,::1024], a_hist[1,::1024], 'o', markevery=0.05, ls=' ', marker=marker, mfc='none', mec=col, markersize=8, label=label) # todo: add log-log plot
        ode_sol = solveLeadOrderODE(phyp, (a_hist[0,0], a_hist[0,-1]))
        ax1.plot(ode_sol.t, ode_sol.y[0], '-', color=col)
        speed = (a_hist[1,1:] - a_hist[1,:-1]) / (a_hist[0,1:] - a_hist[0,:-1])
        a = a_hist[1,1:]
        ax2.plot(a[:1024:-1024], speed[:1024:-1024] / phyp.eps, markevery = 0.05,
                 ls = ' ', marker=marker, mfc='none', mec=col, markersize = 8, label=label)
        # calculate and plot the theoretical prediction
        a_app = 3*phyp.vol/(a**2*(1+1/phyp.gamma[0]))
        b_app = -a_app / phyp.gamma[0]
        b_til = np.sqrt(phyp.gamma[0]) / (np.sqrt(phyp.gamma[0]) + np.sqrt(phyp.gamma[1])) * b_app
        adot_0 = ((a_app - b_til)**3 - phyp.theta_Y**3) / 3
        a_est = phyp.eps * adot_0
        ax2.plot(a[::1024], a_est[::1024] / phyp.eps, '-', color=col)
    # plot the classical reference
    a = np.linspace(1.0, 2.0, 129)
    a_class = phyp.eps * ((3*phyp.vol/a**2)**3 - phyp.theta_Y**3) / 3
    ax2.plot(a, a_class / phyp.eps, 'k--', label="$\\gamma_1 = +\\infty$")
    ax1.set_xlabel("$t$"); ax1.set_ylabel("$a(t)$"); ax1.legend(); 
    ax2.set_xlabel("$a$"); ax2.set_ylabel("$\\dot{a}(t)/\\epsilon$"); ax2.legend()
    return ax1, ax2

def plotCLSpeed_2(cp_list, markers, colors, group_size):
    npzdata = []; params = []; 
    # load the checkpoints and the parameters
    for cp in cp_list:
        parts = cp.split("/")
        npzdata.append(np.load(cp))
        with open(pjoin(*parts[:-1], "PhysicalParameters"), "rb") as f:
            params.append(pickle.load(f))
            params[-1].eps = -1.0 / np.log(params[-1].slip)
    _, ax2 = pyplot.subplots()
    i = 0
    for npz, phyp, marker, col in zip(npzdata, params, markers, colors):
        print("Plotting", phyp)
        label = "$C_b = {:.2f}\\epsilon^2$".format(phyp.bm / phyp.eps**2) if i < group_size else None # print C_b
        a_hist = npz["a_hist"]
        speed = (a_hist[1,1:] - a_hist[1,:-1]) / (a_hist[0,1:] - a_hist[0,:-1])
        a = a_hist[1,1:]
        ax2.plot(a[:1024:-1024], speed[:1024:-1024] / phyp.eps, markevery = ((i % 3) * 0.016, 0.05), 
                 ls = ' ', marker=marker, mfc='none', mec=col, markersize = 8, label=label)
        i += 1
    ax2.set_xlabel("$a$"); ax2.set_ylabel("$\\dot{a}(t)/\\epsilon$"); ax2.legend()
    return ax2

@dataclass
class plotProfileOption:
    plotProfileLines: bool = True
    plotProfileMarkers: bool = False
    plotOuter: bool = False
    plotBending: bool = False

def plotProfiles(test_name, cp_list, opt: plotProfileOption) -> None:
    # load the parameters
    with open(pjoin("result", test_name, "PhysicalParameters"), "rb") as f:
        phyp = pickle.load(f)
    phyp.eps = -1.0 / np.log(phyp.slip)
    print("Parameters of the profiles:", phyp)
    # load the data
    npzdata = []
    for cp in cp_list:
        name = pjoin("result", test_name, "{:04}.npz".format(cp))
        npzdata.append(np.load(name))
    #
    _, ax1 = pyplot.subplots() # axis for plotting the profiles and the asymptotics
    if not opt.plotProfileLines:
        _, ax2 = pyplot.subplots() # axis for plotting the slope dh
        _, ax3 = pyplot.subplots() # axis for plotting the slope dg
    alpha_list = np.linspace(1.0, 0.4, len(cp_list))[::-1]
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
        if opt.plotProfileLines:
            ax1.plot(a * xi_c[2:-2], g[1:-1], '-', color="tab:orange", label=label, alpha=alpha)
            ax1.plot(a * xi_c[2:2+n_fluid], h[2:-1], '-', color="tab:blue", alpha=alpha)
            continue
        if opt.plotProfileMarkers:
            ax1.plot(a * xi_c[2:-2:4], g[1:-1:4], ls='none', marker="o", mfc='none', mec="tab:orange", label=label, alpha=alpha)
            ax1.plot(a * xi_c[2:2+n_fluid:4], h[2:-1:4], ls='none', marker="o", mfc='none', mec="tab:blue", alpha=alpha)
        # calculate and plot the slopes
        z = np.log(a * (1.0 - xi_c[2:n_fluid+2])) * phyp.eps + 1.0
        dh = (h[3:n_fluid+3] - h[1:n_fluid+1]) / (xi_c[3:n_fluid+3] - xi_c[1:n_fluid+1]) / a
        dg = (g[2:n_fluid+2] - g[:n_fluid]) / (xi_c[3:n_fluid+3] - xi_c[1:n_fluid+1]) / a
        ax2.plot(z, -dh, markevery=0.025, ls='none', marker='o', mfc='none', mec="tab:blue", label=label, alpha=alpha)
        ax3.plot(z, -dg, markevery=0.025, ls='none', marker='o', mfc='none', mec="tab:orange", label=label, alpha=alpha)
        # plot the theoretical prediction
        plotPrediction(xi_c[2:2+n_fluid], xi_c[2:-2], phyp, a, adot[-1], alpha, ax1, ax2, ax3, cp == cp_list[-1], opt)
    ax1.legend(); ax1.set_xlabel("$x$"); ax1.set_xlim(0.0, 2.5); ax1.set_ylabel("$z$")
    if not opt.plotProfileLines:
        ax2.legend(); ax2.set_xlabel("$s$"); ax2.set_ylabel("$\partial_{y}h$")
        ax3.legend(); ax3.set_xlabel("$s$"); ax3.set_ylabel("$\partial_{y}g$")

# these are the specific solutions appeared in the next-order solution of the bending problem
def varphi(x: np.ndarray) -> np.ndarray:
    r1 = np.exp(x) * expi(-x)
    r2 = np.exp(-x) * expi(x)
    r = (r1 - r2) / 2
    r[x == 0.0] = 0.0
    return r

def dvarphi(x: np.ndarray) -> np.ndarray:
    r1 = np.exp(x) * expi(-x)
    r2 = np.exp(-x) * expi(x)
    r = (r1 + r2) / 2
    r[x == 0.0] = 0.0
    return r

def plotPrediction(xi_f: np.ndarray, xi_s: np.ndarray, phyp: PhysicalParameters, 
                   a: float, adot: float, alpha: float, ax1, ax2, ax3, addLabels: bool, opt: plotProfileOption) -> None:
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
        if opt.plotOuter:
            ax1.plot(xf, h0 + eps * h1, 'k-', label="Asymptotics" if addLabels else None)
            ax1.plot(xs, g0 + eps * g1, 'k-')
        y = a - xf
        z = np.log(y) * eps + 1.0
        dh_outer = a_app * xf / a + eps * a0 / th_app**2 * (np.log(y) + (3-np.log(2*a)))
        mask = z > 0.7
        ax2.plot(z[mask], dh_outer[mask], 'k-', label="Outer" if addLabels else None)

        # ================== Inner region ==================
        s = y / phyp.slip
        dh_inner = th_y + b_til + eps * a0 / th_y * (np.log(th_y*s + 1) / th_y + s * np.log(1 + 1/(th_y*s)) + 1)
        mask = z < 0.2
        ax2.plot(z[mask], dh_inner[mask], 'm--', label="Inner" if addLabels else None)

        # ================== Intermediate region ==================
        m0 = (th_y**3 + 3*a0*z)**(1/3)
        m01 = m0 + 1 / m0**2 * (ea1*z + eps * a0 * (th_y + np.log(th_y) + 1))
        mask = (z > 0.0) & (z < 0.5)
        ax2.plot(z[mask], m01[mask] + b_til, 'm-', label="Intermediate" if addLabels else None)
        
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
                        D[0]*lb[0]**2*np.exp(-y1) - b_app/(2*a)*y_til**2 + a0*lb[0]/(gamma[0]*th_app**2)*(varphi(y1) - y1*np.log(y1) + (E1+1)*y1) + F1, 
                        D[1]*lb[1]**2*np.exp(y_til / lb[1]))
        #
        h_til_0 = a_app * y_til + C2
        h_til_1 = a0/th_app**2 * (y_til * np.log(y_til) + (2 + np.log(eps/(2*a)))*y_til) - a_app/(2*a)*y_til**2 + D[1]*lb[1]
        if opt.plotBending:
            ax1.plot(xs, (h_til_0 + eps * h_til_1) * eps, 'k-', label="Asymptotics" if addLabels else None)
            ax1.plot(xs, (g_til_0 + eps * g_til_1) * eps, 'k-')
        # plot the slopes
        y_til = y_til[y_til > 0.0]
        y1 = y1[y1 > 0.0]
        z = np.log(y_til * eps) * eps + 1.0
        mask = (z > 0.3) & (z < 0.7)
        dg_til_0 = b_app - C1 / lb[0] * np.exp(-y1)
        dg_til_1 = -D[0]*lb[0]*np.exp(-y1) - b_app/a*y_til + a0/(gamma[0]*th_app**2)*(dvarphi(y1) - np.log(y1) + E1)
        ax3.plot(z, dg_til_0 + eps * dg_til_1, 'k-', label="Asymptotics" if addLabels else None)
        dh_til_0 = a_app
        dh_til_1 = a0/th_app**2*(np.log(y_til) + 3 + np.log(eps/(2*a))) - a_app/a*y_til
        ax2.plot(z[mask], (dh_til_0 + eps*dh_til_1)[mask], 'k--', label="Bending" if addLabels else None) # y_til -> +infty    

if __name__ == "__main__":

    pyplot.rc("font", size=16)

    # getTimeConvergence()
    # getSpaceConvergence()

    # =================================================================
    if True:
        markers = "o^sX"
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        cp_list = ["result/tf/tf-s-4-g{}-Y1-B0-adv/0064.npz".format(i) for i in (1, 2, 4, 8)]
        ax1, ax2 = plotCLSpeed(cp_list, markers, colors)
        ax2.set_xlim(1.0, 1.8); ax2.set_ylim(0, 6) # for spreading

        cp_list = ["result/tf/tf-s-4-g{}-Y1-rec/0064.npz".format(i) for i in (1, 2, 4, 8)]
        ax1, ax2 = plotCLSpeed(cp_list, markers, colors)
        ax2.set_xlim(1.5, 1.98); ax2.set_ylim(-0.4, 0) # for receding
        
        cp_list.clear()
        for g, b in product((1, 2, 4, 8), (-2, -1, 0)):
            cp_list += ["result/tf/tf-s-4-g{}-Y1-B{}-adv/0064.npz".format(g, b)]
        markers = "o^s" * 4
        colors = ["tab:blue"] * 3 + ["tab:orange"] * 3 + ["tab:green"] * 3 + ["tab:red"] * 3
        ax2 = plotCLSpeed_2(cp_list, markers, colors, 3)
        ax2.set_xlim(1.0, 1.8); ax2.set_ylim(0, 6) # for spreading

    # =================================================================
    if False:
        opt = plotProfileOption()
        # plotProfiles("tf-s-4-g2-Y1-B0-adv", [1, 4, 16, 64], opt)
        # plotProfiles("tf-s-4-g2-Y1-rec", [1, 8, 24, 48], opt)

        opt.plotProfileLines = False
        opt.plotProfileMarkers = True
        opt.plotOuter = True
        plotProfiles("tf-s-4-g2-Y1-B0-adv", [2, 8], opt)
        opt.plotOuter = False; opt.plotBending = True
        plotProfiles("tf-s-4-g2-Y1-B0-adv", [2, 8], opt)
        plotProfiles("tf-s-4-g2-Y1-rec", [8], opt)

    pyplot.show()
