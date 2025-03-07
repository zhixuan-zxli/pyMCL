import numpy as np
from matplotlib import pyplot

if __name__ == '__main__':
    # Generate data
    nu = np.concatenate((np.linspace(-0.5, -0.4, 65), np.linspace(-0.4, 1.0, 65)[1:]))
    sigma = nu * np.sqrt(1 + 2*nu)
    sigma_l = nu * (2+3*nu) / (2*np.sqrt(1+2*nu))

    # Plot data
    pyplot.rc("font", size=16)
    pyplot.plot(nu, sigma, '-', label='$\\sigma$')
    pyplot.plot(nu[1:], sigma_l[1:], '--', label='$\\sigma_{\Lambda}$')
    pyplot.plot((-1/3, -1/3), (-1.0, sigma_l[-1]), 'k--') # the reference line of x=-1/3
    pyplot.xlabel('$\\nu$')
    pyplot.gca().set_xlim(-0.5, 1.0)
    pyplot.gca().set_ylim(-1.0, sigma_l[-1])
    pyplot.legend()
    pyplot.savefig("Cauchy_stress.eps", bbox_inches="tight")
    pyplot.show()
