"""Loader for CelebA-HQ"""

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class UnlabeledImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root=root, transform=transform)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image

    def __len__(self):
        return len(self.dataset)


def get_celeba_hq_loader(
    batch_size: int, data_root: str, num_workers: int = 0, img_size: int = 256
):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # Normalize to [-1, 1]
        ]
    )

    dataset = UnlabeledImageFolder(root=data_root, transform=transform)
    assert len(dataset) == 30_000, "Expected 30,000 images in CelebA-HQ"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
