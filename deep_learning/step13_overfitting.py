import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(42)

transform = transforms.ToTensor()
train_data = torchvision.datasets.MNIST(root="./mnist_data", train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root="./mnist_data", train=False, download=True, transform=transform)

# use only a small slice of training data on purpose -- overfitting shows up
# much faster with less data, since there's less variety to force generalization
small_train = torch.utils.data.Subset(train_data, range(1000))

train_loader = torch.utils.data.DataLoader(small_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)

# a big, over-powered MLP for only 1000 training images: plenty of
# capacity to memorize every single one of them individually
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 10),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def accuracy(loader):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


for epoch in range(30):
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
