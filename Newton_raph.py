import numpy as np
import matplotlib.pyplot as plt 

def newton_method(f, df, d2f, x0, tol=1e-6, max_iter=100):
    """
    Newton's method for optimization.

    Parameters:
    - f: Objective function.
    - df: First derivative of the objective function.
    - d2f: Second derivative of the objective function.
    - x0: Initial guess.
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.

    Returns:
    - x_opt: Optimal solution.
    - iter_count: Number of iterations.
    """

    x_opt = x0
    iter_count = 0

    while iter_count < max_iter:
        f_prime = df(x_opt)
        f_double_prime = d2f(x_opt)

        if abs(f_prime) < tol:
            break

        x_opt = x_opt - f_prime / f_double_prime
        iter_count += 1

    return x_opt, iter_count

# Example usage:
def objective_function(x):
    return x**3 - 4*x**2 + 2

def derivative(x):
    return 3 * x**2 - 8*x

def second_derivative(x):
    return 6*x - 8

initial_guess = 3.0
optimal_solution, iterations = newton_method(objective_function, derivative, second_derivative, 
                                             initial_guess)

print("Optimal Solution:", optimal_solution)
print("Number of Iterations:", iterations)

x_vals = np.linspace(-4, 4, 100)
y_vals = objective_function(x_vals)

# Plot the function
plt.plot(x_vals, y_vals, label='Objective Function')

# Plot the iterates of Newton's method
x_optimal, iterations = newton_method(objective_function, derivative, second_derivative, initial_guess)
plt.scatter(x_optimal, objective_function(x_optimal), color='red', label='Optimal Solution')
plt.plot(x_optimal, objective_function(x_optimal), 'ro')

plt.title("Newton's Method Optimization")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
