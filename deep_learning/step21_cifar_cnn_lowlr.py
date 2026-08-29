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
train_eval_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_data, range(5000)), batch_size=1000)

# same architecture as step17/step19 -- no dropout, so this run isolates the
# learning rate as the only change from the original (unstable) CNN runs
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

loss_fn = nn.CrossEntropyLoss()
BASE_LR = 3e-4  # was 1e-3 in every earlier CIFAR CNN run
optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)


def accuracy(loader):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images)
            correct += (preds.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


N_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
print(f"CNN parameters: {sum(p.numel() for p in model.parameters()):,}  base_lr={BASE_LR}", flush=True)

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

    train_acc = accuracy(train_eval_loader)
    test_acc = accuracy(test_loader)
    gap = train_acc - test_acc

    if test_acc > best_acc:
        best_acc, best_epoch = test_acc, epoch
        torch.save(model.state_dict(), "best_cnn_cifar_lowlr.pt")

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"epoch {epoch:2d}  lr={lr_now:.6f}  loss={loss.item():.4f}  "
          f"train acc={train_acc:.2f}%  test acc={test_acc:.2f}%  gap={gap:5.2f}%  ({epoch_time:.1f}s)", flush=True)

print(f"best: epoch {best_epoch}  test accuracy={best_acc:.2f}%", flush=True)
print("DONE", flush=True)
