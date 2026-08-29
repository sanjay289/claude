import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(42)

# each MNIST image is 28x28 grayscale, label 0-9
transform = transforms.ToTensor()  # converts PIL image -> tensor, pixel values scaled to [0, 1]
train_data = torchvision.datasets.MNIST(root="./mnist_data", train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root="./mnist_data", train=False, download=True, transform=transform)

# DataLoader is new: it slices the 60,000 images into random shuffled
# batches of 64 so we're not doing one enormous forward pass at once
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)

# same nn.Sequential pattern as step6, just bigger:
# 784 inputs (28*28 flattened pixels) -> 128 hidden -> 10 outputs (one score per digit)
# ReLU instead of Tanh (the standard choice for hidden layers in bigger nets)
model = nn.Sequential(
    nn.Flatten(),           # 28x28 image -> 784-length vector
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10),      # 10 raw scores, one per digit -- no activation, CrossEntropyLoss handles that
)

# CrossEntropyLoss: the standard loss for "pick 1 of N classes",
# replaces the MSELoss we used for the +1/-1 toy problem
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam instead of plain SGD -- converges faster

for epoch in range(3):
    for images, labels in train_loader:
        preds = model(images)          # 1. forward pass on a batch of 64 images at once
        loss = loss_fn(preds, labels)  # 2. loss

        optimizer.zero_grad()          # 3a. zero grads
        loss.backward()                # 3b. backward pass
        optimizer.step()               # 4. update

    # after each epoch (one full pass over all 60,000 training images),
    # check accuracy on the 10,000 held-out test images the model never trained on
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images)
            predicted_digit = preds.argmax(dim=1)
            correct += (predicted_digit == labels).sum().item()
            total += labels.size(0)

    print(f"epoch {epoch}  loss={loss.item():.4f}  test accuracy={100 * correct / total:.2f}%")
