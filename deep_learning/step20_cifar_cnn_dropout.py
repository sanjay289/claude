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
# fixed 5,000-image slice of the training set, just for cheaply estimating
# train accuracy each epoch -- not used for training itself
train_eval_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_data, range(5000)), batch_size=1000)

# same conv backbone as step17/step19, with Dropout added:
# 0.25 after each pooling stage (lighter, spatial feature maps),
# 0.5 before the final linear layer (heavier, standard for the last FC layer)
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2), nn.Dropout(0.25),

    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2), nn.Dropout(0.25),

    nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2), nn.Dropout(0.25),

    nn.Flatten(),
    nn.Dropout(0.5),
    nn.Linear(64 * 4 * 4, 10),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)


def accuracy(loader):
    model.eval()  # dropout off for evaluation -- unlike step17/19, this now matters
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images)
            correct += (preds.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    model.train()  # dropout back on for training
    return 100 * correct / total


N_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
print(f"CNN parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

best_acc, best_epoch = 0.0, -1
for epoch in range(N_EPOCHS):
    model.train()
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
        torch.save(model.state_dict(), "best_cnn_cifar_dropout.pt")

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"epoch {epoch:2d}  lr={lr_now:.6f}  loss={loss.item():.4f}  "
          f"train acc={train_acc:.2f}%  test acc={test_acc:.2f}%  gap={gap:5.2f}%  ({epoch_time:.1f}s)", flush=True)

print(f"best: epoch {best_epoch}  test accuracy={best_acc:.2f}%", flush=True)
print("DONE", flush=True)
