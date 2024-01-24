import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Define the objective function and its gradient
def objective_function(x):
    return x**2 + 10

def gradient(x):
    return 2*x

# Gradient Descent with Optimal Step (Line Search)
def gradient_descent_optimal_step(initial_param, gradient, objective_function, max_iterations=100, 
                                  tol=1e-6):
    
    """
    Perform gradient descent with optimal step using line search.

    Parameters:
    - initial_param: Initial value of the parameter.
    - gradient: Function to compute the gradient of the objective function.
    - objective_function: Objective function to minimize.
    - max_iterations: Maximum number of iterations.
    - tol: Tolerance for convergence.

    Returns:
    - np.array: Array containing the parameter values at each iteration.
    """

    params = [initial_param]
    for iteration in range(max_iterations):
        grad = gradient(params[-1])
        
        # Use minimize_scalar for line search
        result = minimize_scalar(lambda alpha: objective_function(params[-1] - alpha * grad))
        print(result)
        step_size = result.x

        params.append(params[-1] - step_size * grad)

        if np.linalg.norm(grad) < tol:
            print(f"Converged in {iteration} iterations")
            break
    print("optimal step solution: ",params[-1])
    return np.array(params)

# Gradient Descent with Fixed Step
def gradient_descent_fixed_step(initial_param, gradient, max_iterations=1000, step_size=0.1, tol=1e-6):
    
    """
    Perform gradient descent with fixed step size.

    Parameters:
    - initial_param: Initial value of the parameter.
    - gradient: Function to compute the gradient of the objective function.
    - max_iterations: Maximum number of iterations.
    - step_size: Fixed step size for the gradient descent.
    - tol: Tolerance for convergence.

    Returns:
    - np.array: Array containing the parameter values at each iteration.
    """
    
    params = [initial_param]
    for iteration in range(max_iterations):
        grad = gradient(params[-1])
        params.append(params[-1] - step_size * grad)

        if np.linalg.norm(grad) < tol:
            print(f"Converged in {iteration} iterations")
            break
    print("fixed step solution: ",params[-1])
    return np.array(params)

# Visualization
def plot_optimization_process(params, title):
    x_vals = np.linspace(-5, 5, 100)
    y_vals = objective_function(x_vals)

    plt.plot(x_vals, y_vals, label='Objective Function')
    plt.scatter(params, objective_function(params), color='red', label='Optimization Process')
    plt.title(title)
    plt.xlabel('Parameter Value')
    plt.ylabel('Objective Function Value')
    plt.legend()
    plt.show()

# Perform optimization with optimal step
initial_param = 4.0
params_optimal_step = gradient_descent_optimal_step(initial_param, gradient, objective_function)
print(params_optimal_step)
plot_optimization_process(params_optimal_step, 'Gradient Descent with Optimal Step')

# Perform optimization with fixed step
initial_param = 4.0
params_fixed_step = gradient_descent_fixed_step(initial_param, gradient)
print(params_fixed_step)
plot_optimization_process(params_fixed_step, 'Gradient Descent with Fixed Step')
