import torch
import torch.nn as nn

torch.manual_seed(0)

model_a = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 4 * 4, 10),
)
model_a.eval()

x = torch.randn(1000, 3, 32, 32)  # same size as a CIFAR test batch

with torch.no_grad():
    out_before = model_a(x).clone()

torch.save(model_a.state_dict(), "determinism_test.pt")

model_b = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 4 * 4, 10),
)
model_b.load_state_dict(torch.load("determinism_test.pt"))
model_b.eval()

with torch.no_grad():
    out_after = model_b(x)

# 1. are the weight tensors themselves bit-identical after the save/load round trip?
weights_identical = all(
    torch.equal(pa, pb) for pa, pb in zip(model_a.state_dict().values(), model_b.state_dict().values())
)

# 2. running the SAME already-loaded model_a again on the same input, same process --
#    does even this (no save/load involved at all) reproduce the same output?
with torch.no_grad():
    out_before_again = model_a(x)
same_model_repeat_identical = torch.equal(out_before, out_before_again)

# 3. does the save/load round trip change the output?
reload_output_identical = torch.equal(out_before, out_after)
max_diff = (out_before - out_after).abs().max().item()

print(f"weight tensors bit-identical after save/load: {weights_identical}")
print(f"same live model, called twice, identical output: {same_model_repeat_identical}")
print(f"output identical after save/load round trip: {reload_output_identical}")
print(f"max logit difference after save/load: {max_diff}")
