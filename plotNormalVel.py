import numpy as np
from matplotlib import pyplot

def plotNormalVec(V: float, alpha: float, eta: np.ndarray) -> None:
    # eta_mesh: (x, )
    lab_str = "alpha = {:.3f} * pi".format(alpha/np.pi)
    a = np.cosh(eta) + np.cos(alpha)
    chi = np.sqrt(V * np.sin(alpha)**2 / (alpha - np.sin(alpha)*np.cos(alpha))) # the radius
    chi_dot = 1.
    alpha_dot = chi_dot / chi * np.sin(alpha) * (alpha - np.sin(alpha)*np.cos(alpha)) / (alpha*np.cos(alpha) - np.sin(alpha))
    assert alpha_dot <= 0.
    R = chi / np.sin(alpha)
    R_dot = (chi_dot - R * np.cos(alpha) * alpha_dot) / np.sin(alpha)
    v = (chi_dot / a + chi*alpha_dot*np.sin(alpha) / a**2) * np.hstack((np.sinh(eta), np.broadcast_to(np.sin(alpha), eta.shape)))
    v += chi / a * np.array(((0., np.cos(alpha)*alpha_dot), )) # (x, 2)
    normal = np.hstack((np.sin(alpha) * np.sinh(eta), 1. + np.cos(alpha) * np.cosh(eta))) / a # (x, 2)
    vn = np.sum(v * normal, axis=1) # (x, )
    pyplot.plot(eta, vn, 'o', label=lab_str)
    pyplot.plot(eta, R_dot * (1. - alpha / np.sin(alpha) * (np.cosh(eta) * np.cos(alpha) + 1.) / a), '-', label=lab_str)
    #
    pass

if __name__ == "__main__":
    eta_mesh = np.concatenate((np.linspace(-8., -4., 17)[:-1], np.linspace(-4., 4., 65)[:-1], np.linspace(4., 8., 17)))
    eta_mesh = eta_mesh.reshape(-1, 1)
    pyplot.figure()
    plotNormalVec(np.pi/8, np.pi/2, eta_mesh)
    plotNormalVec(np.pi/8, np.pi/3, eta_mesh)
    # plotNormalVec(np.pi/8, np.pi*2/3, eta_mesh)
    pyplot.legend()
    pyplot.show()
