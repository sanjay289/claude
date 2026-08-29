import torch
import torch.nn as nn

torch.manual_seed(42)

# same architecture as before: 3 -> 4 -> 4 -> 1, tanh activations
# nn.Sequential replaces our hand-written MLP/Layer/Neuron classes;
# nn.Linear replaces the weighted-sum-plus-bias each Neuron computed by hand
model = nn.Sequential(
    nn.Linear(3, 4), nn.Tanh(),
    nn.Linear(4, 4), nn.Tanh(),
    nn.Linear(4, 1), nn.Tanh(),
)

# same dataset, but as tensors -- one matrix of inputs, one vector of targets,
# so all 4 examples are computed in one shot instead of a Python loop over Neurons
xs = torch.tensor([
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
])
ys = torch.tensor([[1.0], [-1.0], [-1.0], [1.0]])

loss_fn = nn.MSELoss(reduction="sum")           # replaces sum((yp - yt) ** 2 ...)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)  # replaces the manual p -= lr * p.grad loop

for step in range(100):
    ypred = model(xs)                # 1. forward pass -- all 4 examples at once
    loss = loss_fn(ypred, ys)        # 2. loss

    optimizer.zero_grad()            # 3a. zero grads (replaces the manual p.grad = None loop)
    loss.backward()                  # 3b. backward pass
    optimizer.step()                 # 4. update (replaces the manual torch.no_grad() loop)

    if step % 10 == 0 or step == 99:
        print(f"step {step:3d}  loss = {loss.item():.4f}")

print()
print("final predictions vs targets:")
final = model(xs)
for pred, target in zip(final, ys):
    print(f"  pred={pred.item():+.4f}   target={target.item():+.1f}")
