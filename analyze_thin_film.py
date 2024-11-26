from math import sqrt
import numpy as np
from fem.post import printConvergenceTable
from thin_film import downsample
import matplotlib
from matplotlib import pyplot
from colorama import Fore, Style

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
    print(Fore.GREEN + "\nTime convergence: " + Style.RESET_ALL)
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
    print(Fore.GREEN + "\nSpace convergence: " + Style.RESET_ALL)
    printConvergenceTable(table_headers, error_table)

def plotContactLine() -> None:
    # dt = 1/(1024*4*8)
    datanames = ["tf-s-4-g2-adap", "tf-s-4-g4-adap", "tf-s-4-g8-adap"]
    data = []
    for name in datanames:
        data.append(np.load("result/" + name + "/0032.npz"))
    # plot the contact line location
    _, ax1 = pyplot.subplots()
    _, ax2 = pyplot.subplots()
    for name, npz in zip(datanames, data):
        a_hist = npz["a_hist"]
        ax1.plot(a_hist[0], a_hist[1], '-', label=name)
        speed = (a_hist[1,1:] - a_hist[1,:-1]) / (a_hist[0,1:] - a_hist[0,:-1])
        ax2.plot(a_hist[0,1:-1], speed[1:], '-', label=name)
    ax1.legend()
    ax2.legend()
    pyplot.show()

def plotSystemTrajectory() -> None:
    filename = "tf-s-4-g4-adap-Y0.4"
    checkpoints = [2, 8, 16, 32]
    npzdata = []
    for cp in checkpoints:
        name = "result/" + filename + "/{:04}.npz".format(cp)
        npzdata.append(np.load(name))
    #
    fig, ax = pyplot.subplots()
    alpha_list = np.linspace(0.2, 1.0, len(checkpoints))
    for data, alpha in zip(npzdata, alpha_list):
        # alpha = 1.0
        xi_c = data["xi_c"]
        a_hist = data["a_hist"]
        h = data["h"]
        g = data["g"]
        t = a_hist[0, -1]
        a = a_hist[1, -1]
        gline = ax.plot(a * xi_c[2:-2], g[1:-1], "-", label=f"t={t}", alpha=alpha)
        n_fluid = h.size - 3
        ax.plot(a * xi_c[2:2+n_fluid], h[2:-1], "-", color=gline[0].get_color(), alpha=alpha)
    ax.legend()
    pyplot.show()
    fig.savefig("thin films.png", dpi=300)

def plotOuter(xf: np.ndarray, xs: np.ndarray, gamma: tuple[float], B: float, V0: float, a: float, ax, alpha: float = 1.0) -> None:
    # solve for p0 from V0
    t1 = a**3*(1/gamma[0]+1/gamma[2])/3
    t2 = a - sqrt(B/gamma[0])*(1 - np.exp(-sqrt(gamma[0]/B)*a))
    c10 = B/(gamma[0]*sqrt(gamma[0]/gamma[1])*(sqrt(gamma[0])+sqrt(gamma[1])))*(-a/sqrt(B)-1/sqrt(gamma[1]))
    p0 = V0 / (t1 + c10*t2)
    # solve for the parameters in the sheet
    c1 = c10 * p0
    c2 = B/(sqrt(gamma[0]*gamma[1])*(sqrt(gamma[0])+sqrt(gamma[1])))*(1/sqrt(gamma[0]) - a/sqrt(B)) * p0
    d1 = c2 - c1 - p0/(2*gamma[0])*a**2
    #
    h = p0/(2*gamma[2])*(a**2-xf**2) + c2
    g = np.where(xs <= a, \
                 p0/(2*gamma[0])*xs**2 + c1*np.exp(sqrt(gamma[0]/B)*(xs-a)) + d1, \
                 c2*np.exp(-sqrt(gamma[1]/B)*(xs-a))
    )
    # check volume
    nf = xf.size
    vh = np.sum((h[1:] + h[:-1])/2*(xf[1:] - xf[:-1]))
    vg = np.sum((g[1:nf] + g[:nf-1])/2*(xs[1:nf] - xs[:nf-1]))
    print("Numerical volume = {:.6f}".format(vh - vg))
    #
    lines = ax.plot(xf, h, '-', alpha=alpha)
    ax.plot(xs, g, '-', color=lines[0].get_color(), alpha=alpha)

def plotAppContactAngle(ax, gamma: np.ndarray, B: float, lb: float, V0: float, a: np.ndarray) -> None:
    if lb != 0.0:
        B = lb**2*gamma[0] # for fixing bending length
    # solve for p0 from V0
    t1 = a**3*(1/gamma[0]+1/gamma[2])/3
    t2 = a - np.sqrt(B/gamma[0])*(1 - np.exp(-np.sqrt(gamma[0]/B)*a))
    c10 = B/(gamma[0]*np.sqrt(gamma[0]/gamma[1])*(np.sqrt(gamma[0])+np.sqrt(gamma[1])))*(-a/np.sqrt(B)-1/np.sqrt(gamma[1]))
    p0 = V0 / (t1 + c10*t2) # type: np.ndarray
    # solve for the apparent contact angles
    dh = -p0*a/gamma[2]
    c2 = B/(np.sqrt(gamma[0]*gamma[1])*(np.sqrt(gamma[0])+np.sqrt(gamma[1])))*(1/np.sqrt(gamma[0]) - a/np.sqrt(B)) * p0
    dg = -c2*np.sqrt(gamma[1]/B)
    #
    if isinstance(a, np.ndarray):
        ax.plot(a, dg-dh, '-', label=r"$\gamma_1={:.2f}, B={:.2e}$".format(gamma[0], B))
        ax.set_xlabel(r"$a$")
    else:
        if lb != 0.0:
            label=r"$l_b={:.2f}, a={:.2f}$".format(lb, a)
        else:
            label=r"$B={:.1e}, a={:.2f}$".format(B, a)
        lines = ax.plot(gamma[0], dg-dh, '-', label=label)
        dh_limit = -3*V0/(gamma[2]*a**2)
        ax.plot((gamma[0,0], gamma[0,-1]), (-dh_limit, -dh_limit), '--', color=lines[0].get_color())
        ax.set_xlabel(r"$\gamma_1$")
    ax.set_ylabel(r"$\theta_{\mathrm{app}}$")
    

if __name__ == "__main__":

    # getTimeConvergence()
    # getSpaceConvergence()
    # plotContactLine()
    plotSystemTrajectory()
    # varying a
    fig, ax = pyplot.subplots()
    V0 = 1.0
    gamma = np.zeros((3, 65))
    gamma[0] = np.linspace(0.5, 20.0, 65)
    gamma[1] = gamma[0] + 0.9
    gamma[2,:] = 1.0
    plotAppContactAngle(ax, gamma, 0.0, 0.1, V0, 1.0)
    plotAppContactAngle(ax, gamma, 0.0, 0.1, V0, 1.2)
    plotAppContactAngle(ax, gamma, 0.0, 0.1, V0, 1.4)
    ax.legend()
    # 
    fig, ax = pyplot.subplots()
    a = np.linspace(1.0, 2.0, 65)
    plotAppContactAngle(ax, (2.0, 2.9, 1.0), 0.0, 0.1, 1.0, a)
    plotAppContactAngle(ax, (4.0, 4.9, 1.0), 0.0, 0.1, 1.0, a)
    plotAppContactAngle(ax, (8.0, 8.9, 1.0), 0.0, 0.1, 1.0, a)
    plotAppContactAngle(ax, (20.0, 20.9, 1.0), 0.0, 0.1, 1.0, a)
    ax.legend()

    pyplot.show()
