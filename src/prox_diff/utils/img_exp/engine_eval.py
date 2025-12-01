"""Compute FID for models, modularized."""

import logging
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from ...core import ProxModel, ProxSamplerVP, ScoreModel, ScoreSamplerVP
from .fid_core import sample_and_compute_fid
from .networks.ddpm_unet.model import UNetTime
from .networks.ddpm_unet.model_prox import get_network
from .vae import get_vae

logger = logging.getLogger(__name__)


def get_fid_name(n_samples):
    if n_samples % 1000 == 0:
        fid_name = f"fid{n_samples // 1000}k"
    else:
        fid_name = f"fid{n_samples}"
    return fid_name


def compute_fid_for_ckpt(
    config, ckpt_path, overwrite=False, method_suffix="", output_root=None
):
    ###########################################################################
    # Unpack config
    ###########################################################################
    stat_npz = config.fid.stat_npz
    model_labels = config.fid.model_labels
    n_steps_list = config.fid.steps
    seed = config.fid.seed
    n_samples = config.fid.n_samples
    batch_size = config.fid.batch_size
    sample_method = config.fid.sample_method

    ###########################################################################
    # Prepare
    ###########################################################################
    device = torch.device("cuda")
    fid_name = get_fid_name(n_samples)
    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt_id = os.path.splitext(os.path.basename(ckpt_path))[0]
    model_keys = {
        "ema0.9999": "ema0.9999_model_state_dict",
        "ema0.999": "ema0.999_model_state_dict",
        "base": "model_state_dict",
    }

    ###########################################################################
    # Run
    ###########################################################################
    for model_label in tqdm(model_labels, desc="Looping over model labels"):
        state_dict_key = model_keys[model_label]
        for n_steps in tqdm(n_steps_list, desc="Looping over steps"):
            save_dir = os.path.join(
                output_root or ckpt_dir,
                f"{fid_name}/{ckpt_id}/{model_label}/{sample_method}{method_suffix}/steps_{n_steps}",
            )
            metrics_file = os.path.join(save_dir, "metrics.json")

            if os.path.exists(metrics_file) and not overwrite:
                logger.info(f"[Skip] Already exists: {metrics_file}")
                continue

            logger.info(f"{save_dir=}")
            _, sampler, vae = build_model_and_sampler(config, ckpt_path, state_dict_key)
            sampler.sample_method = sample_method
            sampler.n_steps = n_steps
            sampler.to(device)
            if vae:
                vae.to(device)

            generator = torch.Generator(device).manual_seed(seed)
            fid = sample_and_compute_fid(
                sampler.forward,
                save_dir,
                n_samples,
                batch_size,
                generator,
                device,
                stat_npz,
                config.fid.save_samples,
                vae=vae,
            )
            logger.info(f"[FID] {save_dir} = {fid:.2f}")


def build_model_and_sampler(config, ckpt_path: str, state_dict_key: str):
    ###########################################################################
    # Unpack config
    ###########################################################################
    model_class = config.model_class
    net_config = config.net
    image_size = config.data.image_size
    channels = config.data.channels
    beta_min = config.diffusion.beta_min
    beta_max = config.diffusion.beta_max

    ###########################################################################
    # Initialize model and sampler
    ###########################################################################
    if model_class == "prox":
        model_type = config.get("model_type", "epsilon")
        # Initialize model
        net = get_network(net_config)
        model = ProxModel(net, model_type=model_type)
        sampler = ProxSamplerVP(model, beta_min, beta_max)

    elif model_class == "score":
        score_sampler_config = {}  # default
        if "score_sampler" in config:
            score_sampler_config = config.score_sampler
        # Initialize model
        net_kwargs = net_config.to_dict()
        net = UNetTime(**net_kwargs)
        model = ScoreModel(net, beta_min, beta_max)
        sampler = ScoreSamplerVP(model, beta_min, beta_max, **score_sampler_config)
        # print(vars(sampler))

    else:
        raise ValueError(f"Unknown model class: {model_class}")

    sampler.data_dim = (channels, image_size, image_size)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sampler.model.load_state_dict(ckpt[state_dict_key])

    ###########################################################################
    # Initialize vae
    ###########################################################################
    vae = None
    if config.get("model", {}).get("use_vae", False):
        vae = get_vae()

    if vae:
        # VAE model uses latent shape
        sampler.data_dim = config.model.vae_latent_shape
    else:
        sampler.data_dim = (channels, image_size, image_size)

    ###########################################################################
    # Convert to DataParallel and eval mode
    ###########################################################################
    sampler.model = nn.DataParallel(sampler.model).eval()
    assert not (sampler.model.training or sampler.model.module.model.training)

    return model, sampler, vae
