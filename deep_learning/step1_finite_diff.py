def f(x):
    return 3 * x**2 - 4 * x + 5

x = 2.0
h = 0.0001

# slope = (change in output) / (change in input)
slope = (f(x + h) - f(x)) / h
print(f"f({x}) = {f(x)}")
print(f"approx derivative at x={x}: {slope}")
