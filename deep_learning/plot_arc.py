"""Plot test-accuracy-per-epoch for the CIFAR-10 arc (steps 16-23).

Reads the *_run.log files already checked into this directory and produces
cifar_arc_comparison.png: one line per step, showing how each change to the
model/training recipe affected the accuracy curve.
"""
import re
import matplotlib.pyplot as plt

RUNS = {
    "step16: MLP baseline": "step16_mlp_run.log",
    "step17: CNN baseline": "step17_cnn_run.log",
    "step20: + dropout": "step20_dropout_run.log",
    "step21: + low LR": "step21_lowlr_run.log",
    "step22: dropout + low LR": "step22_combined_run.log",
    "step23: + augment + BatchNorm": "step23_run.log",
}

EPOCH_RE = re.compile(r"^epoch\s+(\d+).*?test acc(?:uracy)?=([\d.]+)%", re.MULTILINE)


def parse_log(path):
    with open(path) as f:
        text = f.read()
    epochs, accs = [], []
    for m in EPOCH_RE.finditer(text):
        epochs.append(int(m.group(1)))
        accs.append(float(m.group(2)))
    return epochs, accs


def main():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, filename in RUNS.items():
        epochs, accs = parse_log(filename)
        ax.plot(epochs, accs, marker="o", markersize=3, label=label)

    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("CIFAR-10 arc: test accuracy per epoch (steps 16-23)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("cifar_arc_comparison.png", dpi=150)
    print("wrote cifar_arc_comparison.png")


if __name__ == "__main__":
    main()
