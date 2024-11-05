import numpy as np
from fem.post import printConvergenceTable
from thin_film import downsample
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


    

if __name__ == "__main__":
    # getTimeConvergence()
    # getSpaceConvergence()
    # plotContactLine()
    plotSystemTrajectory()