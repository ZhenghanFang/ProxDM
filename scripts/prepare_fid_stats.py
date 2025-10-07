import os
import shutil

import torch
from pytorch_fid.fid_score import save_fid_stats
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from prox_diff.utils.img_exp.fid_core import save_samples

device = torch.device("cuda")

###############################################################################
# Compute training data stat for FID on MNIST
###############################################################################
data_root = "data/mnist"

# Save training data as png
# define image transformations (e.g. using torchvision)
transform = transforms.Compose(
    [
        transforms.Pad(2),  # Pad 2 pixels on all sides to go from 28x28 to 32x32
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ]
)

mnist_train = datasets.MNIST(
    data_root,
    train=True,
    download=True,
    transform=transform,
)

loader = DataLoader(mnist_train, batch_size=1, shuffle=False)

png_folder = os.path.join(data_root, "MNIST_fid/train")
os.makedirs(png_folder, exist_ok=True)
for i, (x, _) in enumerate(tqdm(loader)):
    save_path = os.path.join(png_folder, f"{i}.png")
    save_samples(x, [save_path])


# Compute stat
save_fid_stats(
    [png_folder, os.path.join(data_root, "MNIST_fid/train_stat.npz")],
    batch_size=128,
    device=device,
    dims=2048,
)

os.makedirs("assets/fid_stats", exist_ok=True)
shutil.copy(
    os.path.join(data_root, "MNIST_fid/train_stat.npz"),
    os.path.join("assets/fid_stats/mnist.npz"),
)
shutil.rmtree(png_folder)

###############################################################################
# Compute training data stat for FID on CIFAR10
###############################################################################
data_root = "data/cifar10"
cifar10_train = datasets.CIFAR10(
    data_root,
    train=True,
    download=True,
    transform=None,
)

# Save images as png
png_folder = os.path.join(data_root, "cifar10_fid/train_png")
os.makedirs(png_folder, exist_ok=True)
for i in tqdm(range(len(cifar10_train))):
    img, _ = cifar10_train[i]
    img.save(f"{png_folder}/{i}.png")

# Compute stat for train set
save_fid_stats(
    [png_folder, os.path.join(data_root, "cifar10_fid/train_stat.npz")],
    batch_size=100,
    device=device,
    dims=2048,
)

shutil.copy(
    os.path.join(data_root, "cifar10_fid/train_stat.npz"),
    os.path.join("assets/fid_stats/cifar10.npz"),
)
shutil.rmtree(png_folder)
