# deep_learning

A step-by-step walk from a from-scratch scalar autograd engine up to
convolutional networks on CIFAR-10. Each `stepN_*.py` is a single self-contained
script that changes one thing from the step before it; the `.log` files are the
runs those scripts produced on this device.

## The earlier steps (1–15)

| Steps | What they build |
| --- | --- |
| `step1`–`step4` | From-scratch reverse-mode autograd (`engine.py`) and a hand-built MLP (`nn.py`): finite-difference derivative, then an exact gradient from a topologically-sorted backward pass, then training the MLP on a 4-point toy set. |
| `step5`–`step6` | The same network in PyTorch — first with raw tensors and hand-written `Neuron`/`Layer` classes, then idiomatically with `nn.Sequential` + `nn.Linear`. |
| `step7`–`step8` | First real dataset: an MNIST MLP, then plotting the digits it gets wrong (`misclassified.png`). |
| `step9` | Convolution intuition: a hand-designed 3×3 vertical-edge filter applied to one image (`conv_demo.png`). |
| `step10`–`step11` | An MNIST CNN, then the same CNN trained for more epochs. |
| `step12` | Gradient-descent overshoot: `f(x)=x²` with a well-tuned vs. too-large learning rate. |
| `step13`–`step15` | Overfitting on purpose (tiny MNIST slice), then fixing it with dropout, then adding an `StepLR` schedule to the MNIST CNN. |

## The CIFAR-10 arc (16–23)

CIFAR-10 is the first hard dataset in the series: 32×32 **colour** images of
real objects and backgrounds, not clean centred digits. Steps 16–23 keep the
data fixed and change the model and training recipe one knob at a time.

| Step | Change from previous | Params | Best test acc | Train–test gap (late) |
| --- | --- | --- | --- | --- |
| `step16` | MLP baseline: `3072 → 512 → 512 → 10` | 1,841,162 | **49.97%** (ep 5 of 6) | not tracked |
| `step17` | 3-block CNN (`conv→relu→maxpool` ×3, then one linear) | 66,570 | **58.6%**, **60.9%** on a rerun | not tracked |
| `step18` | Diagnostic only: save/load a fresh CNN and check the weights and outputs survive the round trip | — | — (probe) | — |
| `step19` | Resume `step17`'s checkpoint and keep training to epoch 15 | 66,570 | **57.6%** (ep 6), then diverged to ~35% | — |
| `step20` | Add dropout: 0.25 after each pool, 0.5 before the linear. Start tracking train accuracy on a 5k slice | 66,570 | **56.5%** (ep 4) | 2–5% |
| `step21` | Drop Adam base LR `1e-3 → 3e-4` (no dropout) | 66,570 | **62.9%** (ep 8) | 5–7% |
| `step22` | Dropout **and** LR `3e-4` together | 66,570 | **53.4%** (ep 13) | ~2% |
| `step23` | Add `RandomCrop(32, padding=4)` + `RandomHorizontalFlip` augmentation and `BatchNorm2d` after every conv | 66,890 | **63.2%** (ep 13) | ~1% |

All runs: `torch.manual_seed(42)`, batch size 64, Adam, `StepLR(step_size=4,
gamma=0.5)`, 15 epochs (`step16` and `step17` were run for 6).

### What the arc taught

- **A small CNN beats a big MLP.** `step17` reaches ~60% with **28× fewer
  parameters** than `step16`'s MLP at ~50%. Spatial weight sharing is worth
  far more here than raw parameter count.
- **The learning rate was the main source of instability.** With `lr=1e-3`,
  epochs collapse at random — `step17` epoch 5 fell to 35.9%, and `step19`'s
  resumed run drifted down to ~35% and never recovered. Dropping to `3e-4`
  (`step21`) both raised the ceiling (62.9%) and stopped the collapses. This
  was the single highest-value change in the arc.
- **Dropout didn't help at this scale.** The `step17` model isn't badly
  overfit in 15 epochs, so removing capacity with dropout (`step20`) just
  slowed learning, and stacking it on top of the lower LR (`step22`) was the
  *worst* CNN result in the arc — underfitting, not regularising.
- **Harder data beats less capacity.** `step23`'s augmentation + BatchNorm is
  the only change that closed the train–test gap to ~1% *and* gave the best
  accuracy. Feeding the model shifted/mirrored views each epoch regularises
  without throwing away model capacity.
- **Resuming training is fragile.** `step19` had to hand-patch `initial_lr`
  and `last_epoch` because `StepLR` doesn't reconstruct past decay, and the
  run diverged anyway. A clean restart was more reliable than a resume.

### Caveats when reading the logs

- **`loss=` is the last mini-batch's loss, not an epoch average.** That's why
  it jumps around (1.0–2.8) even in epochs where accuracy climbs smoothly.
  Don't read a trend into it.
- **The LR decays to near-zero by the end.** `gamma=0.5` every 4 epochs over
  15 epochs is 3 halvings — `3e-4` becomes `~3.7e-5` by epoch 12, so the last
  few epochs barely train. The "best epoch" is usually mid-run.
- **Numbers are single-seed, single-run, and noisy.** Epoch times swing from
  ~240s to ~2400s on this device (memory pressure / swap). `step17`'s reload
  check printed 61.46% for weights that scored 60.89% moments earlier —
  consistent with OpenBLAS thread nondeterminism, not a save/load bug.
- **There is no validation split.** Only test accuracy is watched, so "best
  test acc" is mildly optimistic as an estimate of true generalisation.

## Artifacts in this directory

- `best_cnn_cifar*.pt` — the best checkpoint from each CNN run, named per step.
- `conv_demo.png`, `misclassified.png` — figures from `step9` / `step8`.
- `cifar_data/`, `mnist_data/` — downloaded datasets (git-ignored).

## Running

```bash
python3 step23_cifar_cnn_augment_bn.py        # 15 epochs (default)
python3 step23_cifar_cnn_augment_bn.py 20     # optional epoch-count override
```

The from-scratch steps have tiny standalone checks:

```bash
python3 step2_test_engine.py
python3 step3_test_nn.py
```
