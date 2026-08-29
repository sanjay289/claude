import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(42)

transform = transforms.ToTensor()
train_data = torchvision.datasets.MNIST(root="./mnist_data", train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root="./mnist_data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)

# identical architecture to step10/step11 -- only the optimizer setup changes
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3), nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(32 * 5 * 5, 10),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# halve the learning rate every 5 epochs: 1e-3 -> 5e-4 (epoch 5) -> 2.5e-4 (epoch 10)
# same idea as step12's overshoot demo: shrink the step size as training
# progresses so the optimizer can settle into a minimum instead of bouncing out of it
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print(f"CNN parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

N_EPOCHS = 15
best_acc = 0.0
best_epoch = -1
for epoch in range(N_EPOCHS):
    for images, labels in train_loader:
        preds = model(images)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()  # decay the learning rate after each epoch

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images)
            correct += (preds.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    test_acc = 100 * correct / total

    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"epoch {epoch:2d}  lr={lr_now:.6f}  loss={loss.item():.4f}  test accuracy={test_acc:.2f}%", flush=True)

print(f"best: epoch {best_epoch}  test accuracy={best_acc:.2f}%", flush=True)
print("DONE", flush=True)
