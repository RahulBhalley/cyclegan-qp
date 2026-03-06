# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
from contextlib import nullcontext

from networks import Generator, Critic
from config import config
from data import load_data, safe_sampling
from diff_augment import DiffAugment

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories():
    """Ensure all necessary directories exist."""
    dirs_to_make = [config.CKPT_DIR]
    for style in config.STYLE_NAMES:
        dirs_to_make.append(os.path.join(config.CKPT_DIR, style))
    
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)

def train():
    # Setup
    torch.manual_seed(config.RANDOM_SEED)
    torch.set_float32_matmul_precision(config.MATMUL_PRECISION)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    iter_a, iter_b = load_data()

    # Networks
    critic_a = Critic().to(device, memory_format=torch.channels_last)
    critic_b = Critic().to(device, memory_format=torch.channels_last)
    gen_a2b = Generator(upsample=config.UPSAMPLE).to(device, memory_format=torch.channels_last)
    gen_b2a = Generator(upsample=config.UPSAMPLE).to(device, memory_format=torch.channels_last)

    # Compile models if requested
    if config.USE_COMPILE and hasattr(torch, "compile"):
        try:
            logger.info("Compiling models...")
            critic_a = torch.compile(critic_a)
            critic_b = torch.compile(critic_b)
            gen_a2b = torch.compile(gen_a2b)
            gen_b2a = torch.compile(gen_b2a)
        except Exception as e:
            logger.warning(f"Compilation failed: {e}. Proceeding without compilation.")

    # Optimizers
    critic_a_optim = optim.Adam(critic_a.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
    critic_b_optim = optim.Adam(critic_b.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
    gen_a2b_optim = optim.Adam(gen_a2b.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
    gen_b2a_optim = optim.Adam(gen_b2a.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))

    # Mixed Precision Setup
    scaler_c = torch.amp.GradScaler('cuda') if config.USE_AMP and device.type == 'cuda' else None
    scaler_g = torch.amp.GradScaler('cuda') if config.USE_AMP and device.type == 'cuda' else None
    autocast_ctx = torch.amp.autocast('cuda') if config.USE_AMP and device.type == 'cuda' else nullcontext()

    # Load checkpoints
    if config.BEGIN_ITER > 0:
        try:
            networks_to_load = {
                gen_a2b: "gen_a2b",
                gen_b2a: "gen_b2a",
                critic_a: "critic_a",
                critic_b: "critic_b"
            }
            for net, name in networks_to_load.items():
                path = os.path.join(config.CKPT_DIR, config.TRAIN_STYLE, f"{name}_{config.BEGIN_ITER}.pth")
                net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            logger.info(f"Loaded checkpoints from iteration {config.BEGIN_ITER}")
        except Exception as e:
            logger.error(f"Failed to load checkpoints: {e}")

    l1_loss = nn.L1Loss()

    logger.info("Begin training!")
    for i in range(config.BEGIN_ITER, config.END_ITER + 1):
        real_a, real_b = safe_sampling(iter_a, iter_b, device)
        real_a = real_a.to(memory_format=torch.channels_last)
        real_b = real_b.to(memory_format=torch.channels_last)

        #################
        # Train Critics #
        #################
        for param in gen_a2b.parameters(): param.requires_grad = False
        for param in gen_b2a.parameters(): param.requires_grad = False
        for param in critic_a.parameters(): param.requires_grad = True
        for param in critic_b.parameters(): param.requires_grad = True

        for _ in range(2):
            critic_a_optim.zero_grad(set_to_none=True)
            critic_b_optim.zero_grad(set_to_none=True)

            with autocast_ctx:
                with torch.no_grad():
                    fake_b = gen_a2b(real_a)
                    fake_a = gen_b2a(real_b)
                
                real_a_aug = DiffAugment(real_a)
                real_b_aug = DiffAugment(real_b)
                fake_b_aug = DiffAugment(fake_b)
                fake_a_aug = DiffAugment(fake_a)

                score_fake_b = critic_b(fake_b_aug)
                score_real_b = critic_b(real_b_aug)
                score_fake_a = critic_a(fake_a_aug)
                score_real_a = critic_a(real_a_aug)

                loss_val_a = score_real_a - score_fake_a
                if config.NORM == "l1":
                    norm_a = config.LAMBDA * (real_a_aug - fake_a_aug).abs().mean()
                else:
                    norm_a = config.LAMBDA * ((real_a_aug - fake_a_aug)**2).mean().sqrt()
                loss_critic_a = (-loss_val_a + 0.5 * loss_val_a**2 / (norm_a + 1e-8)).mean()

                loss_val_b = score_real_b - score_fake_b
                if config.NORM == "l1":
                    norm_b = config.LAMBDA * (real_b_aug - fake_b_aug).abs().mean()
                else:
                    norm_b = config.LAMBDA * ((real_b_aug - fake_b_aug)**2).mean().sqrt()
                loss_critic_b = (-loss_val_b + 0.5 * loss_val_b**2 / (norm_b + 1e-8)).mean()

                loss_critic_total = loss_critic_a + loss_critic_b

            if scaler_c:
                scaler_c.scale(loss_critic_total).backward()
                scaler_c.step(critic_a_optim)
                scaler_c.step(critic_b_optim)
                scaler_c.update()
            else:
                loss_critic_total.backward()
                critic_a_optim.step()
                critic_b_optim.step()

        ####################
        # Train Generators #
        ####################
        for param in gen_a2b.parameters(): param.requires_grad = True
        for param in gen_b2a.parameters(): param.requires_grad = True
        for param in critic_a.parameters(): param.requires_grad = False
        for param in critic_b.parameters(): param.requires_grad = False

        gen_a2b_optim.zero_grad(set_to_none=True)
        gen_b2a_optim.zero_grad(set_to_none=True)

        with autocast_ctx:
            fake_b = gen_a2b(real_a)
            fake_a = gen_b2a(real_b)
            
            fake_b_aug = DiffAugment(fake_b)
            fake_a_aug = DiffAugment(fake_a)
            real_a_aug = DiffAugment(real_a)
            real_b_aug = DiffAugment(real_b)

            score_fake_b = critic_b(fake_b_aug)
            score_real_b = critic_b(real_b_aug)
            score_fake_a = critic_a(fake_a_aug)
            score_real_a = critic_a(real_a_aug)

            rec_a = gen_b2a(fake_b)
            rec_b = gen_a2b(fake_a)

            loss_adv_a = (score_real_a - score_fake_a).mean()
            loss_adv_b = (score_real_b - score_fake_b).mean()

            loss_cycle_a = l1_loss(rec_a, real_a)
            loss_cycle_b = l1_loss(rec_b, real_b)
            loss_id_a = l1_loss(fake_b, real_b)
            loss_id_b = l1_loss(fake_a, real_a)

            current_cyc_weight = config.CYC_WEIGHT
            if (loss_cycle_a + loss_cycle_b) > 0.5:
                current_cyc_weight *= 1.1

            loss_gen_total = loss_adv_a + loss_adv_b + current_cyc_weight * (loss_cycle_a + loss_cycle_b) + config.ID_WEIGHT * (loss_id_a + loss_id_b)

        if scaler_g:
            scaler_g.scale(loss_gen_total).backward()
            scaler_g.step(gen_a2b_optim)
            scaler_g.step(gen_b2a_optim)
            scaler_g.update()
        else:
            loss_gen_total.backward()
            gen_a2b_optim.step()
            gen_b2a_optim.step()

        if i % config.ITERS_PER_LOG == 0:
            logger.info(f"Iter: {i} | Critic Loss: {loss_critic_total.item():.4f} | Gen Loss: {loss_gen_total.item():.4f}")

        if i % config.ITERS_PER_CKPT == 0:
            ckpt_map = {
                gen_a2b: "gen_a2b",
                gen_b2a: "gen_b2a",
                critic_a: "critic_a",
                critic_b: "critic_b"
            }
            for net, name in ckpt_map.items():
                path = os.path.join(config.CKPT_DIR, config.TRAIN_STYLE, f"{name}_{i}.pth")
                torch.save(net.state_dict(), path)
            logger.info(f"Saved checkpoints at iteration {i}")

if __name__ == "__main__":
    setup_directories()
    train()
