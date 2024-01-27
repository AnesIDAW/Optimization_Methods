import numpy as np
from scipy.optimize import line_search

def quasi_newton_bfgs(obj_func, gradient, x0, tol=1e-6, max_iter=100):
    """
    Quasi-Newton BFGS optimization method for minimizing a scalar objective function.

    Parameters:
    - obj_func (callable): The objective function to be minimized.
    - gradient (callable): The gradient (first derivative) of the objective function.
    - x0 (array_like): Initial guess for the optimal solution.
    - tol (float, optional): Tolerance for stopping criterion. Default is 1e-6.
    - max_iter (int, optional): Maximum number of iterations. Default is 100.

    Returns:
    - x (numpy.ndarray): Optimal solution.
    - f_x (float): Minimum value of the objective function at the optimal solution.
    - iterations (int): Number of iterations performed.
    """
    x = x0
    n = len(x0)
    B = np.eye(n)  # Initial approximation of inverse Hessian
    iterations = 0

    while np.linalg.norm(gradient(x)) > tol and iterations < max_iter:
        p = -np.dot(B, gradient(x))

        # Line search to determine step size
        alpha = line_search(obj_func, gradient, x, p)[0]

        x_next = x + alpha * p
        s = x_next - x
        y = gradient(x_next) - gradient(x)

        # Update B using BFGS formula
        B = B + np.outer(s, s) / np.dot(s, y) - np.dot(np.outer(B @ y, s), B) / np.dot(y, B @ y)

        x = x_next
        iterations += 1

    return x, obj_func(x), iterations

# Example usage:
def objective_function(x):
    return x**2 + 4*x + 4

def gradient_function(x):
    return 2*x + 4

initial_guess = np.array([-5.0])  # Use an array even for a 1D problem
optimal_solution, min_value, iterations = quasi_newton_bfgs(objective_function, gradient_function, 
                                                            initial_guess)

print("Optimal Solution:", optimal_solution)
print("Minimum Value:", min_value)
print("Number of Iterations:", iterations)
