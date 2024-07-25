import numpy as np
from scipy.integrate import quad
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
    vn = R_dot * (1. - alpha / np.sin(alpha) * (np.cosh(eta) * np.cos(alpha) + 1.) / a) # (x, )
    # pyplot.plot(eta, vn, 'o', label=lab_str)
    #
    psi_1 = 2 * np.arctan(((1-np.cos(alpha))/np.sin(alpha)) * np.tanh(eta/2))
    psi_2 = alpha * np.sinh(eta) / (np.cos(alpha) + np.cosh(eta))
    psi = -R_dot/np.sin(alpha) * (psi_1 - psi_2)
    # pyplot.plot(eta, psi, '-', label="psi")
    # pyplot.plot(eta, 1. - alpha/np.sin(alpha)*np.cosh(eta) + 2./np.sin(alpha)*np.arctan((1-np.cos(alpha))/np.sin(alpha)*np.tanh(eta/2)) * np.sinh(eta), 'x')
    # pyplot.plot(eta, (1.0-alpha/np.sin(alpha)) / np.cosh(eta), '--')
    B1 = (np.cos(alpha) - alpha/np.sin(alpha)) / 2.
    B2 = 1 - np.cos(alpha)/2 - alpha/2/np.sin(alpha)
    # pyplot.plot(eta, B1/np.cosh(eta) + B2/np.cosh(eta)**2, '-')

    def integrand(eta, q):
        a = 1. + 2/np.sin(alpha) * np.arctan((1-np.cos(alpha))/np.sin(alpha)*np.tanh(eta/2))*np.sinh(eta) - alpha/np.sin(alpha)*np.cosh(eta)
        return a * np.cos(q*eta)
    q_span = np.concatenate((np.linspace(0., 1., 33)[:-1], np.linspace(1., 8., 33)))
    qK_app = -np.pi**2 * (B1 / np.cosh(np.pi*q_span/2) + B2 * q_span / np.sinh(np.pi*q_span/2))
    qK_num = np.zeros_like(q_span)
    for i, q in enumerate(q_span):
        qK_num[i] = -np.pi * quad(lambda eta: integrand(eta, q), -10., 10.)[0]
    pyplot.plot(q_span, qK_app, '--', label="qK approx")
    pyplot.plot(q_span, qK_num, 'o', label="qK numeric")


if __name__ == "__main__":
    eta_mesh = np.concatenate((np.linspace(-8., -4., 17)[:-1], np.linspace(-4., 4., 65)[:-1], np.linspace(4., 8., 17)))
    eta_mesh = eta_mesh.reshape(-1, 1)
    pyplot.figure()
    plotNormalVec(np.pi/8, np.pi/2, eta_mesh)
    # plotNormalVec(np.pi/8, np.pi/3, eta_mesh)
    # plotNormalVec(np.pi/8, np.pi*2/3, eta_mesh)
    pyplot.legend()
    # pyplot.axis("equal")
    pyplot.show()
