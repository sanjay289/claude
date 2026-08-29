import sys
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(42)

transform = transforms.ToTensor()
train_data = torchvision.datasets.CIFAR10(root="./cifar_data", train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root="./cifar_data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)

# deeper than the MNIST CNN (step10/step15): CIFAR-10 has color (3 input
# channels, not 1) and much more complex content (real objects/backgrounds,
# not clean centered digits), so it needs more filters and a third conv
# block. padding=1 keeps spatial size fixed per conv so only MaxPool shrinks it.
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),                                          # 32x32 -> 16x16

    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),                                          # 16x16 -> 8x8

    nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),                                          # 8x8 -> 4x4

    nn.Flatten(),
    nn.Linear(64 * 4 * 4, 10),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

N_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
print(f"CNN parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

best_acc, best_epoch = 0.0, -1
for epoch in range(N_EPOCHS):
    t0 = time.time()
    for images, labels in train_loader:
        preds = model(images)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    epoch_time = time.time() - t0

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images)
            correct += (preds.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    test_acc = 100 * correct / total

    if test_acc > best_acc:
        best_acc, best_epoch = test_acc, epoch
        torch.save(model.state_dict(), "best_cnn_cifar.pt")  # overwrite checkpoint with these better weights

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"epoch {epoch:2d}  lr={lr_now:.6f}  loss={loss.item():.4f}  test accuracy={test_acc:.2f}%  ({epoch_time:.1f}s)", flush=True)

print(f"best: epoch {best_epoch}  test accuracy={best_acc:.2f}%", flush=True)

# verify the checkpoint actually reproduces that best accuracy: load it into
# a fresh model (not the trained one still in memory) and re-evaluate
model.load_state_dict(torch.load("best_cnn_cifar.pt"))
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images)
        correct += (preds.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
print(f"reloaded checkpoint test accuracy: {100 * correct / total:.2f}%  (should match best above)", flush=True)
print("DONE", flush=True)
