# import sympy as sp
import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot

# def laplacian_in_polar() -> None:
#     # Define the variables
#     r, theta = sp.symbols('r theta')
#     f = sp.Function('f')(r)
#     g = sp.Function('g')(theta)

#     # Define the function F
#     F = f * g

#     # Compute the Laplacian in polar coordinates
#     laplacian_F = (sp.diff(F, r, 2) + (1/r) * sp.diff(F, r) + (1/r**2) * sp.diff(F, theta, 2))

#     # Compute the Laplacian of the Laplacian to get the biharmonic
#     biharmonic_F = (sp.diff(laplacian_F, r, 2) + (1/r) * sp.diff(laplacian_F, r) + (1/r**2) * sp.diff(laplacian_F, theta, 2))

#     # Simplify the result
#     biharmonic_F_simplified = sp.simplify(biharmonic_F)

#     # Display the result
#     print(biharmonic_F_simplified)

def getConvolutionKernel_1(alpha: float, rho_grid: np.ndarray, eps: float = 0.5) -> np.ndarray:
    def integrand(xi, rho):
        s = eps + 1j * xi
        a = np.exp(rho * s) * (np.sin(2*alpha*s) - s * np.sin(2*alpha)) / (2 * s * (np.cos(2*alpha*s) - np.cos(2*alpha)))
        return np.real(a)
    k = np.zeros_like(rho_grid)
    for i, rho in enumerate(rho_grid):
        k[i] = quad(lambda xi: integrand(xi, rho), -8.0*np.pi, 8.0*np.pi)[0]
    return k / (2.0 * np.pi)

def getConvolutionKernel_2(alpha: float, rho_grid: np.ndarray) -> np.ndarray:
    def integrand(rho):
        return np.sinh(rho) / (np.cosh(np.pi*rho/alpha) - 1)
    k = np.zeros_like(rho_grid)
    for i, rho in enumerate(rho_grid):
        k[i] = quad(integrand, abs(rho), 10.0)[0]
    return k * np.pi / (4.0 * alpha**2)

def getConvolutionKernelDer(alpha: float, rho_grid: np.ndarray) -> np.ndarray:
    return -np.pi/(4.0*alpha**2) * np.sinh(rho_grid) / (np.cosh(np.pi*rho_grid/alpha) - 1.0)

def getConvolutionKernelDerByQuad(alpha: float, rho_grid: np.ndarray, eps: float = 0.5) -> np.ndarray:
    def integrand(xi, rho):
        s = eps + 1j * xi
        a = np.exp(rho * s) * (np.sin(2*alpha*s) - s * np.sin(2*alpha)) / (2 * (np.cos(2*alpha*s) - np.cos(2*alpha)))
        return np.real(a)
    k = np.zeros_like(rho_grid)
    for i, rho in enumerate(rho_grid):
        k[i] = quad(lambda xi: integrand(xi, rho), -20., 20.)[0]
    return k / (2.0 * np.pi)    

if __name__ == "__main__":
    # laplacian_in_polar()
    # alpha = np.pi/4
    # rho_grid = np.concatenate((np.linspace(-3., -0.5, 32)[:-1], np.linspace(-0.5, 0.5, 64)[:-1], np.linspace(0.5, 3.0, 32)))
    # k1 = getConvolutionKernel_1(alpha, rho_grid, eps=0.5)
    # k2 = getConvolutionKernel_2(alpha, rho_grid)
    # dk = getConvolutionKernelDer(alpha, rho_grid)
    # dkq = getConvolutionKernelDerByQuad(alpha, rho_grid, eps=0.)
    pyplot.figure()
    # pyplot.plot(rho_grid, k1, 'o-')
    # pyplot.plot(rho_grid, k2, '*-')
    # pyplot.plot(rho_grid, dk, '--')
    # pyplot.plot(rho_grid, dkq, 's-')
    z_neg = np.linspace(-5.0, 0.0, 33)
    z_pos = -z_neg[-1::-1]
    B = -1.0
    pyplot.plot(z_neg, B/2*np.exp(z_neg), '-')
    pyplot.plot(z_pos, B*z_pos + B/2*np.exp(-z_pos), '-')
    pyplot.axis("equal")
    pyplot.show()
