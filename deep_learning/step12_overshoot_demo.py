from engine import Value

# minimize f(x) = x^2 -- a simple bowl, minimum at x=0.
# derivative is 2x, so gradient descent step is: x -= lr * 2x

def run(lr, steps=8):
    x = Value(5.0)
    trace = [x.data]
    for _ in range(steps):
        y = x * x
        x.grad = 0.0
        y.backward()
        x.data -= lr * x.grad
        trace.append(x.data)
    return trace


print("lr=0.1  (well-tuned, curvature-appropriate step):")
for i, v in enumerate(run(0.1)):
    print(f"  step {i}: x = {v:+.4f}")

print()
print("lr=1.1  (too large relative to the bowl's curvature):")
for i, v in enumerate(run(1.1)):
    print(f"  step {i}: x = {v:+.4f}")
