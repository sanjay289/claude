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

# step7's MLP started with nn.Flatten() -- threw away the 2D structure immediately.
# this CNN keeps images 2D through two conv+pool stages before flattening at the end.
model = nn.Sequential(
    # in_channels=1 (grayscale), out_channels=16 (learn 16 different filters), 3x3 filters
    nn.Conv2d(1, 16, kernel_size=3), nn.ReLU(),
    nn.MaxPool2d(2),                              # 26x26 -> 13x13

    nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(),  # 16 input maps -> 32 filters over them
    nn.MaxPool2d(2),                              # 11x11 -> 5x5

    nn.Flatten(),                                 # now flatten: 32 channels x 5 x 5 = 800 numbers
    nn.Linear(32 * 5 * 5, 10),                    # same idea as step7's final layer: 10 class scores
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"CNN parameters: {sum(p.numel() for p in model.parameters()):,}")

for epoch in range(3):
    for images, labels in train_loader:
        preds = model(images)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images)
            correct += (preds.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    print(f"epoch {epoch}  loss={loss.item():.4f}  test accuracy={100 * correct / total:.2f}%")
