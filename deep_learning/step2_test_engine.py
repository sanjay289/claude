from engine import Value

x = Value(2.0)
y = x * x * 3 - x * 4 + 5   # same f(x) = 3x^2 - 4x + 5 as step 1

y.backward()

print(f"y.data = {y.data}")
print(f"x.grad = {x.grad}   (exact, no approximation)")
