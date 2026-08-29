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

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 4 * 4, 10),
)

# resume from the checkpoint saved by step17 (epoch 5 of the previous 6-epoch run)
START_EPOCH = 6
model.load_state_dict(torch.load("best_cnn_cifar.pt"))

loss_fn = nn.CrossEntropyLoss()

# step17's schedule (step_size=4, gamma=0.5, base_lr=1e-3) had already decayed
# once by epoch 5 (lr=5e-4) -- StepLR doesn't recompute decay retroactively
# from last_epoch, so set the *current* rate directly and let last_epoch just
# drive when future boundaries are crossed
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
optimizer.param_groups[0]["initial_lr"] = 1e-3  # original base_lr, for the scheduler's bookkeeping
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5, last_epoch=START_EPOCH - 1)


def evaluate():
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images)
            correct += (preds.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


TOTAL_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
print(f"CNN parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

# re-evaluate the loaded checkpoint fresh, in this process, as the baseline to beat
best_acc = evaluate()
best_epoch = START_EPOCH - 1
print(f"resumed from epoch {best_epoch}  test accuracy={best_acc:.2f}%", flush=True)

for epoch in range(START_EPOCH, TOTAL_EPOCHS):
    t0 = time.time()
    for images, labels in train_loader:
        preds = model(images)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    epoch_time = time.time() - t0

    test_acc = evaluate()

    if test_acc > best_acc:
        best_acc, best_epoch = test_acc, epoch
        torch.save(model.state_dict(), "best_cnn_cifar.pt")

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"epoch {epoch:2d}  lr={lr_now:.6f}  loss={loss.item():.4f}  test accuracy={test_acc:.2f}%  ({epoch_time:.1f}s)", flush=True)

print(f"best: epoch {best_epoch}  test accuracy={best_acc:.2f}%", flush=True)
print("DONE", flush=True)
