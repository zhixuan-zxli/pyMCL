import numpy as np
from matplotlib import pyplot

cp_file = "result/MCL-B0.05-s2/1024.npz"
dt = 1.0/1024/32/4

if __name__ == "__main__":
    cp = np.load(cp_file)
    energy = cp["energy"]
    refcl_hist = cp["refcl_hist"]
    phycl_hist = cp["phycl_hist"]

    t = np.arange(energy.shape[0]) * dt
    pyplot.figure()
    pyplot.plot(t, np.sum(energy, axis=1), '-', label="total")
    pyplot.legend()

    pyplot.figure()
    pyplot.plot(t, refcl_hist[:,0], '-', label="ref left")
    pyplot.plot(t, refcl_hist[:,2], '-', label="ref right")
    pyplot.plot(t, phycl_hist[:,0], '-', label="phy left")
    pyplot.plot(t, phycl_hist[:,2], '-', label="phy right")
    pyplot.legend()

    pyplot.show()
