import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(42)

transform = transforms.ToTensor()
train_data = torchvision.datasets.MNIST(root="./mnist_data", train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root="./mnist_data", train=False, download=True, transform=transform)

small_train = torch.utils.data.Subset(train_data, range(1000))

train_loader = torch.utils.data.DataLoader(small_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)

# identical architecture to step13, with nn.Dropout(0.5) inserted after each
# hidden layer -- the only change from the overfitting run
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(512, 10),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def accuracy(loader):
    model.eval()  # tells Dropout to turn off (use all neurons) for evaluation
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()  # switch dropout back on for training
    return 100 * correct / total


for epoch in range(30):
    model.train()
    for images, labels in train_loader:
        preds = model(images)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 3 == 0 or epoch == 29:
        train_acc = accuracy(train_loader)
        test_acc = accuracy(test_loader)
        gap = train_acc - test_acc
        print(f"epoch {epoch:2d}  train acc={train_acc:6.2f}%  test acc={test_acc:6.2f}%  gap={gap:5.2f}%")
