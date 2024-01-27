# ===========================================
# an implementation of golden section method:
# ===========================================
import math
import matplotlib.pyplot as plt
import numpy as np

def golden_section_search(f, xl, xu, tolerance=1e-5, minimize=True):
   """
   Create a function and try to get the optimum value by using the GR method:
   Exp:
   f = x**2 + 4*x + 5 -> posing xu and xl by gussing
   we have GRatio = (sqrt(5)-1)/2 = 0.61803 ...
  
   calculate D, the biggest interval to shift into opt value:
   D = R * (xu - xl)
  
   Then, calculate the new value of x1 and x2:
   x1 = xl + D
   x2 = xu - D
  
   calculate the tolerance by this equation:
   tolerance =  (1 - R)*abs((xu - xl)/xopt)*100
  
   then we iterate until reaching the desired tolerance.
   """
    R = (math.sqrt(5) - 1) / 2

    while True:
        D = R * (xu - xl)

        x1 = xl + D
        x2 = xu - D

        f1 = f(x1)
        f2 = f(x2)

        if minimize:
            if f1 < f2:
                xu = x2
            else:
                xl = x1
        else:
            if f1 > f2:
                xu = x2
            else:
                xl = x1

        xopt = (xu + xl) / 2
        new_tolerance = (1 - R) * abs((xu - xl) / xopt) * 100

        if new_tolerance < tolerance:
            break

    return xopt

def plot_func(func, optimum_value, xl, xu):
    x_values = np.linspace(xl, xu, 100)
    y_values = func(x_values)

    plt.plot(x_values, y_values, label='Objective Function')
    plt.scatter(optimum_value, func(optimum_value), color='red', label='Optimal Point')

    plt.title('Golden Section Search')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example function: f(x) = x^2 + 4*x + 5
function_exp = lambda x: x**2 + 4 * x + 5

# Initial guesses for xl and xu
xl_initial = -5
xu_initial = 5

# Perform Golden Section Search for minimum
optimal_value_min = golden_section_search(function_exp, xl_initial, xu_initial)

# Perform Golden Section Search for maximum
optimal_value_max = golden_section_search(function_exp, xl_initial, xu_initial, minimize=False)

print("Optimal Value (Minimum):", optimal_value_min)
print("Optimal Value (Maximum):", optimal_value_max)

# Plot the results
plot_func(function_exp, optimal_value_min, xl_initial, xu_initial)
plot_func(function_exp, optimal_value_max, xl_initial, xu_initial)
