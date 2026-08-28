import os
import time
import argparse

import torch
import torchvision as tv
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset

import model.losses as gan_losses
import utils.misc as misc
from model.networks import Generator, Discriminator


# ============================================================
# MONUMENT DATASET
# ============================================================

class MonumentDataset(Dataset):

    def __init__(self, base_dir):

        self.original_dir = os.path.join(
            base_dir, "original"
        )

        self.damaged_dir = os.path.join(
            base_dir, "damaged"
        )

        self.mask_dir = os.path.join(
            base_dir, "masks"
        )

        self.files = sorted([
            f for f in os.listdir(self.original_dir)
            if f.lower().endswith(".png")
        ])

        self.to_tensor = T.ToTensor()

        print(f"Found {len(self.files)} monument images.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        # -------------------------
        # Load original image
        # -------------------------

        original = Image.open(
            os.path.join(
                self.original_dir,
                filename
            )
        ).convert("RGB")

        # -------------------------
        # Load damaged image
        # -------------------------

        damaged = Image.open(
            os.path.join(
                self.damaged_dir,
                filename
            )
        ).convert("RGB")

        # -------------------------
        # Load mask
        # -------------------------

        mask_filename = filename.replace(
            ".png",
            "_mask.png"
        )

        mask = Image.open(
            os.path.join(
                self.mask_dir,
                mask_filename
            )
        ).convert("L")

        # -------------------------
        # Convert to tensors
        # -------------------------

        original = self.to_tensor(original)
        damaged = self.to_tensor(damaged)
        mask = self.to_tensor(mask)

        # -------------------------
        # Convert images
        # [0,1] -> [-1,1]
        # -------------------------

        original = original * 2 - 1
        damaged = damaged * 2 - 1

        # -------------------------
        # Make mask binary
        # -------------------------

        mask = (mask > 0.5).float()

        return original, damaged, mask


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    '--config',
    type=str,
    default="configs/train-monuments.yaml",
    help="Path to yaml config file"
)


# ============================================================
# TRAINING LOOP
# ============================================================

def training_loop(
        generator,
        discriminator,
        g_optimizer,
        d_optimizer,
        gan_loss_g,
        gan_loss_d,
        train_dataloader,
        last_n_iter,
        writer,
        config
):

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        and config.use_cuda_if_available
        else 'cpu'
    )

    print(f"Training device: {device}")

    losses = {}

    generator.train()
    discriminator.train()

    losses_log = {
        'd_loss': [],
        'g_loss': [],
        'ae_loss': [],
        'ae_loss1': [],
        'ae_loss2': [],
    }

    # --------------------------------------------------------
    # Start training
    # --------------------------------------------------------

    init_n_iter = last_n_iter + 1

    train_iter = iter(train_dataloader)

    time0 = time.time()

    for n_iter in range(
        init_n_iter,
        config.max_iters
    ):

        # ====================================================
        # LOAD DATA
        # ====================================================

        try:

            batch_real, batch_incomplete, mask = next(
                train_iter
            )

        except StopIteration:

            train_iter = iter(train_dataloader)

            batch_real, batch_incomplete, mask = next(
                train_iter
            )

        # Move data to GPU

        batch_real = batch_real.to(
            device,
            non_blocking=True
        )

        batch_incomplete = batch_incomplete.to(
            device,
            non_blocking=True
        )

        mask = mask.to(
            device,
            non_blocking=True
        )

        # ====================================================
        # PREPARE INPUT FOR GENERATOR
        # ====================================================

        # One additional channel containing ones

        ones_x = torch.ones_like(
            batch_incomplete[:, 0:1]
        )

        # DeepFill expects:
        #
        # RGB damaged image = 3 channels
        # ones channel       = 1 channel
        # mask channel       = 1 channel
        #
        # Total = 5 channels

        x = torch.cat(
            [
                batch_incomplete,
                ones_x,
                ones_x * mask
            ],
            dim=1
        )

        # ====================================================
        # GENERATOR
        # ====================================================

        x1, x2 = generator(
            x,
            mask
        )

        batch_predicted = x2

        # ====================================================
        # COMPLETE IMAGE
        # ====================================================

        batch_complete = (
            batch_predicted * mask
            +
            batch_incomplete * (1.0 - mask)
        )

        # ====================================================
        # DISCRIMINATOR TRAINING
        # ====================================================

        # Real image + mask

        batch_real_mask = torch.cat(
            (
                batch_real,
                mask
            ),
            dim=1
        )

        # Generated image + mask

        batch_filled_mask = torch.cat(
            (
                batch_complete.detach(),
                mask
            ),
            dim=1
        )

        # Put real and generated images together

        batch_real_filled = torch.cat(
            (
                batch_real_mask,
                batch_filled_mask
            ),
            dim=0
        )

        # Discriminator prediction

        d_real_gen = discriminator(
            batch_real_filled
        )

        d_real, d_gen = torch.split(
            d_real_gen,
            batch_real.size(0)
        )

        # Discriminator loss

        losses['d_loss'] = gan_loss_d(
            d_real,
            d_gen
        )

        # Update discriminator

        d_optimizer.zero_grad()

        losses['d_loss'].backward()

        d_optimizer.step()

        # ====================================================
        # GENERATOR TRAINING
        # ====================================================

        # Stage 1 reconstruction loss

        losses['ae_loss1'] = (
            config.l1_loss_alpha
            *
            torch.mean(
                torch.abs(
                    batch_real - x1
                )
            )
        )

        # Stage 2 reconstruction loss

        losses['ae_loss2'] = (
            config.l1_loss_alpha
            *
            torch.mean(
                torch.abs(
                    batch_real - x2
                )
            )
        )

        # Total reconstruction loss

        losses['ae_loss'] = (
            losses['ae_loss1']
            +
            losses['ae_loss2']
        )

        # Add mask channel for discriminator

        batch_gen = torch.cat(
            (
                batch_complete,
                mask
            ),
            dim=1
        )

        # Discriminator evaluates generated image

        d_gen = discriminator(
            batch_gen
        )

        # GAN generator loss

        losses['g_loss'] = gan_loss_g(
            d_gen
        )

        losses['g_loss'] = (
            config.gan_loss_alpha
            *
            losses['g_loss']
        )

        # Add reconstruction loss

        if config.ae_loss:

            losses['g_loss'] += (
                losses['ae_loss']
            )

        # Update generator

        g_optimizer.zero_grad()

        losses['g_loss'].backward()

        g_optimizer.step()

        # ====================================================
        # LOGGING
        # ====================================================

        for k in losses_log.keys():

            losses_log[k].append(
                losses[k].item()
            )

        # ----------------------------------------------------
        # Print losses
        # ----------------------------------------------------

        if n_iter % config.print_iter == 0:

            dt = time.time() - time0

            print(
                f"\n@iter: {n_iter}: "
                f"{(config.print_iter / dt):.4f} it/s"
            )

            time0 = time.time()

            for k, loss_log in losses_log.items():

                if len(loss_log) == 0:
                    continue

                loss_log_mean = (
                    sum(loss_log)
                    /
                    len(loss_log)
                )

                print(
                    f"{k}: {loss_log_mean:.4f}"
                )

                if config.tb_logging:

                    writer.add_scalar(
                        f"losses/{k}",
                        loss_log_mean,
                        global_step=n_iter
                    )

                loss_log.clear()

        # ====================================================
        # SAVE IMAGES TO TENSORBOARD
        # ====================================================

        if (
            config.tb_logging
            and config.save_imgs_to_tb_iter
            and n_iter % config.save_imgs_to_tb_iter == 0
        ):

            viz_images = [
                misc.pt_to_image(batch_complete),
                misc.pt_to_image(x1),
                misc.pt_to_image(x2)
            ]

            img_grids = [
                tv.utils.make_grid(
                    images[:config.viz_max_out],
                    nrow=2
                )
                for images in viz_images
            ]

            writer.add_image(
                "Inpainted",
                img_grids[0],
                global_step=n_iter,
                dataformats="CHW"
            )

            writer.add_image(
                "Stage 1",
                img_grids[1],
                global_step=n_iter,
                dataformats="CHW"
            )

            writer.add_image(
                "Stage 2",
                img_grids[2],
                global_step=n_iter,
                dataformats="CHW"
            )

        # ====================================================
        # SAVE IMAGES TO DISK
        # ====================================================

        if (
            config.save_imgs_to_disc_iter
            and n_iter % config.save_imgs_to_disc_iter == 0
        ):

            viz_images = [
                misc.pt_to_image(batch_real),
                misc.pt_to_image(batch_complete)
            ]

            img_grids = [
                tv.utils.make_grid(
                    images[:config.viz_max_out],
                    nrow=2
                )
                for images in viz_images
            ]

            tv.utils.save_image(
                img_grids,
                f"{config.checkpoint_dir}/images/iter_{n_iter}.png",
                nrow=2
            )

        # ====================================================
        # SAVE CHECKPOINT
        # ====================================================

        if (
            n_iter % config.save_checkpoint_iter == 0
            and n_iter > init_n_iter
        ):

            misc.save_states(
                "states.pth",
                generator,
                discriminator,
                g_optimizer,
                d_optimizer,
                n_iter,
                config
            )

        # ====================================================
        # BACKUP CHECKPOINT
        # ====================================================

        if (
            config.save_cp_backup_iter
            and n_iter % config.save_cp_backup_iter == 0
            and n_iter > init_n_iter
        ):

            misc.save_states(
                f"states_{n_iter}.pth",
                generator,
                discriminator,
                g_optimizer,
                d_optimizer,
                n_iter,
                config
            )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parser.parse_args()

    config = misc.get_config(
        args.config
    )

    # --------------------------------------------------------
    # Random seed
    # --------------------------------------------------------

    if config.random_seed != False:

        torch.manual_seed(
            config.random_seed
        )

        torch.cuda.manual_seed_all(
            config.random_seed
        )

        import numpy as np

        np.random.seed(
            config.random_seed
        )

    # --------------------------------------------------------
    # Create checkpoint directory
    # --------------------------------------------------------

    if not os.path.isdir(
        config.checkpoint_dir
    ):

        os.makedirs(
            os.path.abspath(
                config.checkpoint_dir
            )
        )

        os.makedirs(
            os.path.abspath(
                f"{config.checkpoint_dir}/images"
            )
        )

        print(
            f"Created checkpoint_dir folder: "
            f"{config.checkpoint_dir}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = MonumentDataset(
        "monument_damage_dataset"
    )

    print(
        f"Dataset size: {len(train_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        and config.use_cuda_if_available
        else 'cpu'
    )

    print(
        f"Using device: {device}"
    )

    # --------------------------------------------------------
    # Construct networks
    # --------------------------------------------------------

    cnum_in = config.img_shapes[2]

    generator = Generator(
        cnum_in=cnum_in + 2,
        cnum_out=cnum_in,
        cnum=48,
        return_flow=False
    )

    discriminator = Discriminator(
        cnum_in=cnum_in + 1,
        cnum=64
    )

    generator = generator.to(device)

    discriminator = discriminator.to(device)

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------

    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config.g_lr,
        betas=(
            config.g_beta1,
            config.g_beta2
        )
    )

    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.d_lr,
        betas=(
            config.d_beta1,
            config.d_beta2
        )
    )

    # --------------------------------------------------------
    # GAN loss
    # --------------------------------------------------------

    if config.gan_loss == 'hinge':

        gan_loss_d = gan_losses.hinge_loss_d
        gan_loss_g = gan_losses.hinge_loss_g

    elif config.gan_loss == 'ls':

        gan_loss_d = gan_losses.ls_loss_d
        gan_loss_g = gan_losses.ls_loss_g

    else:

        raise NotImplementedError(
            f"Unsupported loss: {config.gan_loss}"
        )

    # --------------------------------------------------------
    # Load pretrained / previous checkpoint
    # --------------------------------------------------------

    last_n_iter = -1

    if config.model_restore != '':

        state_dicts = torch.load(
            config.model_restore,
            map_location=device
        )

        generator.load_state_dict(
            state_dicts['G']
        )

        if 'D' in state_dicts.keys():

            discriminator.load_state_dict(
                state_dicts['D']
            )

        if 'G_optim' in state_dicts.keys():

            g_optimizer.load_state_dict(
                state_dicts['G_optim']
            )

        if 'D_optim' in state_dicts.keys():

            d_optimizer.load_state_dict(
                state_dicts['D_optim']
            )

        if 'n_iter' in state_dicts.keys():

            last_n_iter = state_dicts['n_iter']

        print(
            f"Loaded models from: "
            f"{config.model_restore}!"
        )

    # --------------------------------------------------------
    # TensorBoard
    # --------------------------------------------------------

    writer = None

    if config.tb_logging:

        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(
            config.log_dir
        )

    # --------------------------------------------------------
    # Start training
    # --------------------------------------------------------

    training_loop(
        generator,
        discriminator,
        g_optimizer,
        d_optimizer,
        gan_loss_g,
        gan_loss_d,
        train_dataloader,
        last_n_iter,
        writer,
        config
    )


# ============================================================
# RUN
# ============================================================

if __name__ == '__main__':

    main()