import random
from nn import MLP
from engine import Value

random.seed(42)

# toy dataset: 4 points in 3D, each labeled +1.0 or -1.0
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired outputs

model = MLP(3, [4, 4, 1])

for step in range(100):
    # 1. forward pass
    ypred = [model(x) for x in xs]

    # 2. loss: mean squared error between prediction and target
    loss = sum((yp - yt) ** 2 for yp, yt in zip(ypred, ys))

    # 3. backward pass -- must zero grads first, they accumulate (+=)
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # 4. update: step every parameter opposite its gradient
    learning_rate = 0.05
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if step % 10 == 0 or step == 99:
        print(f"step {step:3d}  loss = {loss.data:.4f}")

print()
print("final predictions vs targets:")
for x, yt in zip(xs, ys):
    print(f"  pred={model(x).data:+.4f}   target={yt:+.1f}")
