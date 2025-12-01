"""Training function, using hf's accelerate"""

import copy
import os

import torch
from accelerate import Accelerator
from tqdm import tqdm

from ...core import ProxModel, ProxTrainerVP, ScoreModel, ScoreTrainerVP
from ..misc import has_nan_in_torch_model
from .dataloaders.loader_factory import get_loader
from .dataloaders.utils import infinite_loop
from .loss_lr_schedule import PMLossSchedule, ScoreLossLRSchedule
from .networks.ddpm_unet.model import UNetTime
from .networks.ddpm_unet.model_prox import get_network
from .t_lamb_sampling import get_sample_t_lamb_func
from .t_sampling import get_sample_t_func
from .utils import ema, save_ckpt
from .vae import get_vae


def save_config_as_json(config, path: str):
    json_str = config.to_json(indent=4)
    with open(path, "w") as f:
        f.write(json_str)


def train(config, ckpt_dir: str, debug: bool = False):
    ###########################################################################
    # Build accelerator
    ###########################################################################
    accelerator = Accelerator()

    device = accelerator.device
    n_gpus = accelerator.state.num_processes
    accelerator.print(f"Number of GPUs: {n_gpus}")

    ###########################################################################
    # Unpack config
    ###########################################################################
    # Training
    total_iters = config.train.total_iters
    save_every = config.train.save_every
    ema_decay = config.train.ema_decay
    batch_size = config.train.batch_size // n_gpus
    resume = config.train.get("resume", None)
    grad_clip = config.train.get("grad_clip", None)

    # Data
    dataset_name = config.data.dataset
    num_workers = config.data.num_workers

    ###########################################################################
    # Prepare
    ###########################################################################
    os.makedirs(ckpt_dir, exist_ok=True)
    # Save config as json
    save_config_as_json(config, f"{ckpt_dir}/config.json")

    ###########################################################################
    # Data
    ###########################################################################
    loader = get_loader(dataset_name, batch_size, num_workers)

    ###########################################################################
    # Build model, trainer, etc.
    ###########################################################################
    model, trainer, loss_lr_schedule, vae = build_model_trainer_schedule(config)
    # print number of model parameters
    accelerator.print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    if vae:
        accelerator.print(
            f"Number of VAE parameters: {sum(p.numel() for p in vae.parameters()):,}"
        )

    optimizer = torch.optim.AdamW(model.parameters())
    ema_models = {decay: copy.deepcopy(model) for decay in ema_decay}
    train_losses = []
    start_iter = 1

    ###########################################################################
    # Resume from checkpoint if specified
    ###########################################################################
    if resume:
        accelerator.print(f"Resuming from {resume}")
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        train_losses = ckpt["train_losses"]
        for decay, ema_model in ema_models.items():
            ema_model.load_state_dict(ckpt[f"ema{decay}_model_state_dict"])
        start_iter = int(os.path.basename(resume).split(".")[0].split("_")[1]) + 1
        assert start_iter == len(train_losses) + 1

    ###########################################################################
    # Train
    ###########################################################################

    # Prepare
    trainer, optimizer, loader = accelerator.prepare(trainer, optimizer, loader)
    for decay, ema_model in ema_models.items():
        ema_models[decay].to(device)
    if vae:
        vae.to(device)

    def train_step(loss_params):
        model.train()
        x0 = next(loader)
        if vae:
            x0 = vae.encode(x0)
        optimizer.zero_grad()
        loss = trainer(x0, loss_params).mean()
        accelerator.backward(loss)

        if grad_clip is not None:
            norm = accelerator.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        assert not has_nan_in_torch_model(model)
        for decay, ema_model in ema_models.items():
            ema(model, ema_model, decay)
        return loss.item()

    pbar = tqdm(
        range(start_iter, total_iters + 1),
        desc="Training",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    loader = infinite_loop(loader)
    for it in pbar:
        # Get loss parameters and learning rate for current iteration
        loss_params, lr = loss_lr_schedule.get(it)
        set_lr(optimizer, lr)

        train_loss = train_step(loss_params)
        train_losses.append(train_loss)

        if accelerator.is_main_process:
            lr_ = [f"{param_group['lr']:.1e}" for param_group in optimizer.param_groups]
            pbar.set_postfix(
                {"loss": f"{train_loss:.3f}", "loss_params": loss_params, "lr": lr_}
            )

        if accelerator.is_main_process and (
            debug or it % save_every == 0 or it == total_iters
        ):
            # Save model
            ckpt_path = f"{ckpt_dir}/ckpt_{it}.pt"
            save_ckpt(
                ckpt_path,
                model_state_dict=accelerator.unwrap_model(model).state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                train_losses=train_losses,
                **{
                    f"ema{decay}_model_state_dict": ema_model.state_dict()
                    for decay, ema_model in ema_models.items()
                },
            )
            print(f"Saved checkpoint to {ckpt_path}")

    accelerator.wait_for_everyone()


def set_lr(optimizer, lr):
    for p in optimizer.param_groups:
        p["lr"] = lr


def build_model_trainer_schedule(config):
    ###########################################################################
    # Unpack config
    ###########################################################################
    model_class = config.model_class
    net_config = config.net
    beta_min = config.diffusion.beta_min
    beta_max = config.diffusion.beta_max
    total_iters = config.train.total_iters

    ###########################################################################
    # Initialize model and trainer
    ###########################################################################
    if model_class == "prox":
        #######################################################################
        # Unpack prox specific config
        #######################################################################
        # Prox matching loss
        l1_iters = config.train.pm.l1_iters
        pm_gamma_start = config.train.pm.pm_gamma_start
        pm_gamma_decay = config.train.pm.pm_gamma_decay  # multiplicative decay factor
        pm_gamma_decay_stages = (
            config.train.pm.pm_gamma_decay_stages
        )  # number of loss_sigma values throughout training
        l1_lr = config.train.pm.l1_lr
        pm_lr = config.train.pm.pm_lr
        gamma_scaling = config.train.pm.get("gamma_scaling", False)

        # Prox model
        model_type = config.get("model_type", "epsilon")  # default: "epsilon"

        # Prox trainer
        loss_on = config.train.get("loss_on", "epsilon")  # default: "epsilon"

        #######################################################################
        # Initialize model
        #######################################################################
        net = get_network(net_config)
        model = ProxModel(net, model_type=model_type)
        trainer = ProxTrainerVP(model, beta_min, beta_max, loss_on=loss_on)
        trainer.sample_t_lamb = get_sample_t_lamb_func(config)

        loss_lr_schedule = PMLossSchedule(
            total_iters=total_iters,
            l1_iters=l1_iters,
            pm_loss_sigma_start=pm_gamma_start,
            pm_loss_sigma_decay=pm_gamma_decay,
            pm_loss_sigma_decay_stages=pm_gamma_decay_stages,
            l1_lr=l1_lr,
            pm_lr=pm_lr,
            gamma_scaling=gamma_scaling,
        )
    elif model_class == "score":
        # Unpack score specific config
        lr = config.train.lr.base
        warmup = config.train.lr.warmup
        # Initialize model
        net_kwargs = net_config.to_dict()
        net = UNetTime(**net_kwargs)
        model = ScoreModel(net, beta_min, beta_max)
        trainer = ScoreTrainerVP(model, beta_min, beta_max)
        trainer.sample_t = get_sample_t_func(config)

        loss_lr_schedule = ScoreLossLRSchedule(lr, warmup)
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    ###########################################################################
    # Initialize vae
    ###########################################################################
    vae = None
    if config.get("model", {}).get("use_vae", False):
        vae = get_vae()

    return model, trainer, loss_lr_schedule, vae
