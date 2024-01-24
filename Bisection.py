import numpy as np
import matplotlib.pyplot as plt

def bisection_method(func, a, b, tolerance=1e-6, max_iterations=100):
    """
    Bisection Method for finding roots of a function.

    Parameters:
    - func: The function to find the root of.
    - a, b: Initial interval [a, b] where the root lies.
    - tolerance: The desired tolerance for the root.
    - max_iterations: Maximum number of iterations.

    Returns:
    - float: Approximate root.
    - int: Number of iterations performed.
    """

    iteration = 0
    interval_list = [(a, b)]

    while (b - a) / 2 > tolerance and iteration < max_iterations:
        c = (a + b) / 2
        iteration += 1

        if func(c) == 0 or (b - a) / 2 < tolerance:
            break

        if np.sign(func(c)) == np.sign(func(a)):
            a = c
        else:
            b = c

        interval_list.append((a, b))

    return (a + b) / 2, iteration, interval_list

# Example function: f(x) = x^2 - 4
def example_function(x):
    return x**2 - 4

# Plotting the function
x_values = np.linspace(-3, 3, 100)
y_values = example_function(x_values)

plt.plot(x_values, y_values, label='f(x) = x^2 - 4')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8, label='y=0')

# Adjusted initial interval to avoid the ValueError
a, b = -2, 2

# Run the Bisection Method
root, iterations, interval_list = bisection_method(example_function, a, b)

# Plotting the root and intervals
plt.scatter(root, 0, color='red', label='Approximate Root')
for i, (interval_a, interval_b) in enumerate(interval_list, start=1):
    plt.axvline(interval_a, linestyle='--', color='gray', linewidth=0.8)
    plt.axvline(interval_b, linestyle='--', color='gray', linewidth=0.8, 
                label=f'Iteration {i}')

plt.title('Bisection Method - Iterations')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

# Display result
print(f"Approximate root: {root}")
print(f"Iterations: {iterations}")
