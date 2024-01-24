import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

def lagrange_multiplier_critical_points():
    # Define the constraint equation
    constraint_eq = lambda x, y: 4 * x**2 + y**2 - 9
    
    # Define the objective function
    objective_func = lambda x, y: 81 * x**2 + y**2

    # Define the Lagrangian function
    lagrangian = lambda variables, lambd: (
        objective_func(*variables) - lambd * constraint_eq(*variables)
    )
    
    # Define the constraint function for minimization
    constraint = lambda variables: constraint_eq(*variables)
    
    # Initial guess for the variables
    initial_guess = [0, 0]
    
    # Solve the optimization problem
    result = minimize(lagrangian, initial_guess, args=(1,), 
                      constraints={'type': 'eq', 'fun': constraint})
    
    # Extract the critical points
    critical_points = [tuple(result.x)]
    
    return critical_points

def plot_graphs(constraint_eq, critical_points, x_values):
    y_values_positive = constraint_eq(x_values, 0)
    y_values_negative = -constraint_eq(x_values, 0)

    plt.figure(figsize=(4, 4))
    plt.plot(x_values, y_values_positive, label=r'$4x^2 + y^2 = 9$')
    plt.plot(x_values, y_values_negative, linestyle='dashed')

    for point in critical_points:
        plt.scatter(*point, color='red', marker='o')

    t_values = np.linspace(0, 2 * np.pi, 100)
    x_boundary = 3/2 * np.cos(t_values)
    y_boundary = np.sqrt(5) * np.sin(t_values)
    plt.plot(x_boundary, y_boundary, label='Boundary points', linestyle='dashed')

    plt.title('Graph of Constraint and Critical Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()

def plot_3d_graph(x_values, constraint_eq, f, critical_points):
    t_values = np.linspace(0, 2 * np.pi, 100)
    x_boundary = 3/2 * np.cos(t_values)
    y_boundary = np.sqrt(5) * np.sin(t_values)

    X, Y = np.meshgrid(x_values, np.linspace(-3, 3, 400))
    Z = f(X, Y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, label=r'$81x^2 + y^2$')

    ax.plot(x_boundary, y_boundary, np.zeros_like(x_boundary), color='red',
            linestyle='dashed', label=r'$4x^2 + y^2 = 9$')

    for point in critical_points:
        ax.scatter(*point, color='black', s=50, label='Critical Point', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of the Function and Constraint')
    ax.legend()

    plt.show()

# Example usage:
critical_points = lagrange_multiplier_critical_points()
print("Critical Points:", critical_points)

x_values = np.linspace(-3, 3, 400)
plot_graphs(lambda x, y: 4 * x**2 + y**2 - 9, critical_points, x_values)
plot_3d_graph(x_values, lambda x, y: 81 * x**2 + y**2, 
              lambda x, y: 81 * x**2 + y**2, critical_points)
