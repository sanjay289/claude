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

# same shape of model as step7's MNIST MLP, just bigger input:
# 3*32*32=3072 flattened pixels (3 color channels now, not 1) -> 512 -> 512 -> 10 classes
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * 32 * 32, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 10),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

N_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
print(f"MLP parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

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

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"epoch {epoch:2d}  lr={lr_now:.6f}  loss={loss.item():.4f}  test accuracy={test_acc:.2f}%  ({epoch_time:.1f}s)", flush=True)

print(f"best: epoch {best_epoch}  test accuracy={best_acc:.2f}%", flush=True)
print("DONE", flush=True)
