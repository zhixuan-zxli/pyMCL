import sympy as sp

def laplacian_in_polar() -> None:
    # Define the variables
    r, theta = sp.symbols('r theta')
    f = sp.Function('f')(r)
    g = sp.Function('g')(theta)

    # Define the function F
    F = f * g

    # Compute the Laplacian in polar coordinates
    laplacian_F = (sp.diff(F, r, 2) + (1/r) * sp.diff(F, r) + (1/r**2) * sp.diff(F, theta, 2))

    # Compute the Laplacian of the Laplacian to get the biharmonic
    biharmonic_F = (sp.diff(laplacian_F, r, 2) + (1/r) * sp.diff(laplacian_F, r) + (1/r**2) * sp.diff(laplacian_F, theta, 2))

    # Simplify the result
    biharmonic_F_simplified = sp.simplify(biharmonic_F)

    # Display the result
    print(biharmonic_F_simplified)

if __name__ == "__main__":
    laplacian_in_polar()
