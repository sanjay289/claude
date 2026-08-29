import random
from nn import MLP

random.seed(42)

# 3 inputs -> hidden layer of 4 -> hidden layer of 4 -> 1 output
model = MLP(3, [4, 4, 1])

x = [2.0, 3.0, -1.0]
out = model(x)

print(f"input:  {x}")
print(f"output: {out.data}")
print(f"total parameters: {len(model.parameters())}")
