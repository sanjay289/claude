import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
test_data = torchvision.datasets.MNIST(root="./mnist_data", train=False, download=True, transform=transform)

image, label = test_data[1]  # a single 28x28 image, shape (1, 28, 28)

# a hand-designed 3x3 vertical-edge-detecting filter:
# strongly positive on the left, strongly negative on the right ->
# fires (large output) wherever pixel brightness changes left-to-right
vertical_edge_kernel = torch.tensor([
    [1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0],
])


def convolve_by_hand(img, kernel):
    """Slide `kernel` over `img`, computing a dot product at each position.
    This is literally the definition of convolution -- no torch magic yet."""
    k = kernel.shape[0]
    h, w = img.shape
    out = torch.zeros(h - k + 1, w - k + 1)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = img[i:i + k, j:j + k]
            out[i, j] = (patch * kernel).sum()   # the dot product
    return out


img_2d = image.squeeze()          # drop the channel dim -> plain 28x28
by_hand = convolve_by_hand(img_2d, vertical_edge_kernel)

# now the same computation via torch's built-in conv2d, to confirm it matches
# conv2d expects shape (batch, channels, height, width)
img_batched = image.unsqueeze(0)                     # (1, 1, 28, 28)
kernel_batched = vertical_edge_kernel.view(1, 1, 3, 3)  # (out_channels, in_channels, kh, kw)
via_torch = F.conv2d(img_batched, kernel_batched).squeeze()

print(f"digit shown: {label}")
print(f"max difference between hand-rolled and torch conv2d: {(by_hand - via_torch).abs().max().item():.6f}")

fig, axes = plt.subplots(1, 2, figsize=(7, 4))
axes[0].imshow(img_2d, cmap="gray")
axes[0].set_title(f"original (digit {label})")
axes[0].axis("off")
axes[1].imshow(by_hand, cmap="gray")
axes[1].set_title("after vertical-edge filter")
axes[1].axis("off")
plt.tight_layout()
plt.savefig("conv_demo.png", dpi=120)
print("saved conv_demo.png")
