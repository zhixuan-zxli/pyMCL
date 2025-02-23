import numpy as np
from matplotlib import pyplot

a = 1.0

def exact_solution(x, t):
    return np.cos(np.pi * (x - a * t))

def testLW(x: np.ndarray, dt: float, Te: float):
    # prepare for visualization
    pyplot.ion()
    ax = pyplot.subplot()    
    # Set up the initial condition
    u = exact_solution(x, 0.0)
    u_new = np.zeros_like(u)
    # Time stepping loop
    t = 0.0
    while t < Te:
        # Compute the new solution
        adt = a * dt
        u_new[:] = u
        u_new[1:-1] -= adt * ((u[1:-1]-u[:-2])*(1/(x[1:-1]-x[:-2])-1/(x[2:]-x[:-2])) + (u[2:]-u[1:-1])*(1/(x[2:]-x[1:-1])-1/(x[2:]-x[:-2])))
        u_new[1:-1] += adt**2 * ((u[2:]-u[1:-1])/(x[2:]-x[1:-1]) - (u[1:-1]-u[:-2])/(x[1:-1]-x[:-2])) / (x[2:]-x[:-2])
        # Update the solution
        u[:] = u_new[:]
        u[0] = exact_solution(x[0], t)
        u[-1] = exact_solution(x[-1], t)
        t += dt
        # visualize
        ax.clear()
        ax.plot(x, u, 'o')
        ax.plot(x, exact_solution(x, t))
        ax.set_ylim(-1.0, 1.0)
        ax.set_title("t = {:.3f}".format(t))
        pyplot.draw()
        pyplot.pause(0.05)

if __name__ == "__main__":
    # set up the grid. 
    m = 4
    x = np.concatenate((
        np.linspace(0.0, 0.5, m+1), 
        np.linspace(1/2, 3/4, m+1)[1:], 
        np.linspace(3/4, 1.0, 2*m+1)[1:],
    ))
    testLW(x, 1/128, 4.0)