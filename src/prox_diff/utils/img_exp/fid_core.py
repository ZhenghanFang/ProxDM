"""Utils for computing FID."""

import json
import os
import shutil

import matplotlib.pyplot as plt
import torch
import torchvision.utils
from pytorch_fid.fid_score import calculate_fid_given_paths
from tqdm import tqdm


@torch.no_grad()
def sample_by_chunks(sample_func, n_samples, batch_size, save_dir, generator, vae):
    chunks = [batch_size] * (n_samples // batch_size)
    chunks += [n_samples % batch_size] if n_samples % batch_size > 0 else []
    print(f"{chunks=}")
    for i in tqdm(range(len(chunks)), desc="Chunk"):
        for x in sample_func(chunks[i], generator):
            pass
        if vae:
            # Using hf diffuser's AutoencoderKL with DataParallel has issues.
            # So we use single GPU and decode one-by-one to avoid out-of-memory.
            # A better way is to use DistributedDataParallel.
            x_new = []
            for j in tqdm(range(x.shape[0]), desc="Decoding with VAE"):
                x_new.append(vae.decode(x[j].unsqueeze(0)).cpu())
            x = torch.cat(x_new, dim=0)
        x_0 = x.cpu()

        save_paths = [
            f"{save_dir}/{file_i}.png"
            for file_i in range(sum(chunks[:i]), sum(chunks[: i + 1]))
        ]
        print("Saving samples...")
        save_samples(x_0, save_paths)


def save_samples(samples: torch.Tensor, paths: list[str]):
    """Save image samples to files.

    Args:
        samples (torch.Tensor): shape (b, 3 or 1, w, h), range [-1, 1]
        paths (list of str): file paths to save
    """
    assert samples.shape[0] == len(paths)

    samples = samples * 0.5 + 0.5  # convert to [0, 1]

    # tqdm(zip(samples, paths), desc="Saving samples", total=len(paths))
    for x, path in zip(samples, paths):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchvision.utils.save_image(x, path)


def sample_and_compute_fid(
    sample_func,
    save_dir,
    n_samples,
    batch_size,
    generator,
    device,
    stat_npz,
    save_samples=0,
    vae=None,
):
    """Sample and compute FID.
    Samples are saved in save_dir/samples.
    FID is saved in save_dir/metrics.json.

    Args:
        sample_func (callable):
            Inputs:
                - n (int): number of samples
                - generator (torch.Generator): random number generator
            Yields:
                torch.Tensor: samples at each sampling step.
        save_samples: Number of samples to save to disk.
            - If 0, deletes the sample directory.
            - If an integer > 0, keeps only the first n samples.
            - If "all", keeps all generated samples.
    """
    samples_dir = f"{save_dir}/samples"
    sample_by_chunks(sample_func, n_samples, batch_size, samples_dir, generator, vae)

    print("Computing FID...")
    fid = calculate_fid_given_paths(
        paths=[samples_dir, stat_npz], batch_size=100, device=device, dims=2048
    )

    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump({"FID": fid}, f, indent=4)

    if save_samples == "all":
        pass
    elif isinstance(save_samples, int):
        if save_samples == 0:
            shutil.rmtree(samples_dir)
        else:
            # Keep only the first n samples
            all_samples = os.listdir(samples_dir)
            all_samples.sort(key=lambda x: int(x.split(".")[0]))
            for sample in all_samples[save_samples:]:
                os.remove(os.path.join(samples_dir, sample))
    else:
        raise ValueError(f"Invalid save_samples: {save_samples}")

    return fid
