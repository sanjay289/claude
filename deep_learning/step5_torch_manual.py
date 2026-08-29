import random
import torch

random.seed(42)
torch.manual_seed(42)


class Neuron:
    def __init__(self, n_inputs):
        # same as engine.py's Neuron, but Value(...) -> torch.tensor(..., requires_grad=True)
        self.w = [torch.tensor(random.uniform(-1, 1), requires_grad=True) for _ in range(n_inputs)]
        self.b = torch.tensor(random.uniform(-1, 1), requires_grad=True)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return torch.tanh(act)

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_inputs, layer_sizes):
        sizes = [n_inputs] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(layer_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ---- exact same dataset and training loop as step4_train.py ----
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])

for step in range(100):
    # 1. forward pass
    ypred = [model(x) for x in xs]

    # 2. loss
    loss = sum((yp - yt) ** 2 for yp, yt in zip(ypred, ys))

    # 3. backward pass -- torch also accumulates grads, so zero first
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    loss.backward()

    # 4. update -- torch.no_grad() so this manual update isn't itself tracked for grad
    learning_rate = 0.05
    with torch.no_grad():
        for p in model.parameters():
            p -= learning_rate * p.grad

    if step % 10 == 0 or step == 99:
        print(f"step {step:3d}  loss = {loss.item():.4f}")

print()
print("final predictions vs targets:")
for x, yt in zip(xs, ys):
    print(f"  pred={model(x).item():+.4f}   target={yt:+.1f}")
