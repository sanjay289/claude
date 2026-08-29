import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

torch.manual_seed(42)

transform = transforms.ToTensor()
train_data = torchvision.datasets.MNIST(root="./mnist_data", train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root="./mnist_data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# same training as step7, just no per-epoch printout this time
for epoch in range(3):
    for images, labels in train_loader:
        preds = model(images)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ---- find misclassified examples ----
model.eval()
wrong_images, wrong_preds, wrong_labels = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images).argmax(dim=1)
        mismatch = preds != labels
        wrong_images.append(images[mismatch])
        wrong_preds.append(preds[mismatch])
        wrong_labels.append(labels[mismatch])

wrong_images = torch.cat(wrong_images)
wrong_preds = torch.cat(wrong_preds)
wrong_labels = torch.cat(wrong_labels)

print(f"total misclassified: {len(wrong_images)} / {len(test_data)}")

# ---- plot the first 12 ----
n = 12
fig, axes = plt.subplots(3, 4, figsize=(8, 7))
for i, ax in enumerate(axes.flat):
    img = wrong_images[i].squeeze()
    ax.imshow(img, cmap="gray")
    ax.set_title(f"pred: {wrong_preds[i].item()}  true: {wrong_labels[i].item()}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("misclassified.png", dpi=120)
print("saved misclassified.png")
