# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
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
    dirs_to_make = []
    if config.TRAIN:
        dirs_to_make.append(config.CKPT_DIR)
        for style in config.STYLES:
            dirs_to_make.append(os.path.join(config.CKPT_DIR, style))
    else:
        dirs_to_make.append(config.SAMPLE_DIR)
        for style in config.STYLES:
            style_dir = os.path.join(config.SAMPLE_DIR, style)
            dirs_to_make.extend([
                style_dir,
                os.path.join(style_dir, config.OUT_STY_DIR),
                os.path.join(style_dir, config.OUT_REC_DIR)
            ])
    
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)

def train():
    # Setup
    torch.manual_seed(config.RANDOM_SEED)
    torch.set_float32_matmul_precision(config.MATMUL_PRECISION)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    X_data, Y_data = load_data()

    # Networks
    C_X = Critic().to(device, memory_format=torch.channels_last)
    C_Y = Critic().to(device, memory_format=torch.channels_last)
    G = Generator(upsample=config.UPSAMPLE).to(device, memory_format=torch.channels_last)
    F = Generator(upsample=config.UPSAMPLE).to(device, memory_format=torch.channels_last)

    # Compile models if requested
    if config.USE_COMPILE and hasattr(torch, "compile"):
        try:
            logger.info("Compiling models...")
            C_X = torch.compile(C_X)
            C_Y = torch.compile(C_Y)
            G = torch.compile(G)
            F = torch.compile(F)
        except Exception as e:
            logger.warning(f"Compilation failed: {e}. Proceeding without compilation.")

    # Optimizers
    C_X_optim = optim.Adam(C_X.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
    C_Y_optim = optim.Adam(C_Y.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
    G_optim = optim.Adam(G.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
    F_optim = optim.Adam(F.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))

    # Mixed Precision Setup
    scaler_c = torch.amp.GradScaler('cuda') if config.USE_AMP and device.type == 'cuda' else None
    scaler_g = torch.amp.GradScaler('cuda') if config.USE_AMP and device.type == 'cuda' else None
    autocast_ctx = torch.amp.autocast('cuda') if config.USE_AMP and device.type == 'cuda' else nullcontext()

    # Load checkpoints
    if config.BEGIN_ITER > 0:
        try:
            for net, name in zip([G, F, C_X, C_Y], ["G", "F", "C_X", "C_Y"]):
                path = os.path.join(config.CKPT_DIR, config.TRAIN_STYLE, f"{name}_{config.BEGIN_ITER}.pth")
                net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            logger.info(f"Loaded checkpoints from iteration {config.BEGIN_ITER}")
        except Exception as e:
            logger.error(f"Failed to load checkpoints: {e}")

    l1_loss = nn.L1Loss()

    logger.info("Begin training!")
    for i in range(config.BEGIN_ITER, config.END_ITER + 1):
        x_real, y_real = safe_sampling(X_data, Y_data, device)
        x_real = x_real.to(memory_format=torch.channels_last)
        y_real = y_real.to(memory_format=torch.channels_last)

        #################
        # Train Critics #
        #################
        for param in G.parameters(): param.requires_grad = False
        for param in F.parameters(): param.requires_grad = False
        for param in C_X.parameters(): param.requires_grad = True
        for param in C_Y.parameters(): param.requires_grad = True

        for _ in range(2):
            C_X_optim.zero_grad(set_to_none=True)
            C_Y_optim.zero_grad(set_to_none=True)

            with autocast_ctx:
                with torch.no_grad():
                    # X -> Y
                    G_x = G(x_real)
                    # Y -> X
                    F_y = F(y_real)
                
                # Apply DiffAugment to both real and fake
                # This makes the critic task harder and prevents overfitting on small datasets
                x_real_aug = DiffAugment(x_real)
                y_real_aug = DiffAugment(y_real)
                G_x_aug = DiffAugment(G_x)
                F_y_aug = DiffAugment(F_y)

                C_Y_G_x = C_Y(G_x_aug)
                C_y_y = C_Y(y_real_aug)
                C_X_F_y = C_X(F_y_aug)
                C_X_x = C_X(x_real_aug)

                # QP-div loss calculation
                x_loss_val = C_X_x - C_X_F_y
                if config.NORM == "l1":
                    x_norm = config.LAMBDA * (x_real_aug - F_y_aug).abs().mean()
                else:
                    x_norm = config.LAMBDA * ((x_real_aug - F_y_aug)**2).mean().sqrt()
                x_loss = (-x_loss_val + 0.5 * x_loss_val**2 / (x_norm + 1e-8)).mean()

                y_loss_val = C_y_y - C_Y_G_x
                if config.NORM == "l1":
                    y_norm = config.LAMBDA * (y_real_aug - G_x_aug).abs().mean()
                else:
                    y_norm = config.LAMBDA * ((y_real_aug - G_x_aug)**2).mean().sqrt()
                y_loss = (-y_loss_val + 0.5 * y_loss_val**2 / (y_norm + 1e-8)).mean()

                c_loss = x_loss + y_loss

            if scaler_c:
                scaler_c.scale(c_loss).backward()
                scaler_c.step(C_X_optim)
                scaler_c.step(C_Y_optim)
                scaler_c.update()
            else:
                c_loss.backward()
                C_X_optim.step()
                C_Y_optim.step()

        ####################
        # Train Generators #
        ####################
        for param in G.parameters(): param.requires_grad = True
        for param in F.parameters(): param.requires_grad = True
        for param in C_X.parameters(): param.requires_grad = False
        for param in C_Y.parameters(): param.requires_grad = False

        G_optim.zero_grad(set_to_none=True)
        F_optim.zero_grad(set_to_none=True)

        with autocast_ctx:
            G_x = G(x_real)
            F_y = F(y_real)
            
            # Use augmented fakes for adversarial loss
            G_x_aug = DiffAugment(G_x)
            F_y_aug = DiffAugment(F_y)
            # Use augmented reals for adversarial consistency
            x_real_aug = DiffAugment(x_real)
            y_real_aug = DiffAugment(y_real)

            C_Y_G_x = C_Y(G_x_aug)
            C_y_y = C_Y(y_real_aug)
            C_X_F_y = C_X(F_y_aug)
            C_X_x = C_X(x_real_aug)

            F_G_x = F(G_x)
            G_F_y = G(F_y)

            # Adversarial losses
            x_adv_loss = (C_X_x - C_X_F_y).mean()
            y_adv_loss = (C_y_y - C_Y_G_x).mean()

            # Cycle-consistency & Identity
            x_cyc_loss = l1_loss(F_G_x, x_real)
            y_cyc_loss = l1_loss(G_F_y, y_real)
            x_id_loss = l1_loss(G_x, y_real)
            y_id_loss = l1_loss(F_y, x_real)

            # Dynamic Loss Balancing (Simple version: adjust weights if cycle loss is too high)
            # This helps the generator prioritize content preservation if it's failing
            cyc_weight = config.CYC_WEIGHT
            if (x_cyc_loss + y_cyc_loss) > 0.5: # Threshold example
                cyc_weight *= 1.1

            g_loss = x_adv_loss + y_adv_loss + cyc_weight * (x_cyc_loss + y_cyc_loss) + config.ID_WEIGHT * (x_id_loss + y_id_loss)

        if scaler_g:
            scaler_g.scale(g_loss).backward()
            scaler_g.step(G_optim)
            scaler_g.step(F_optim)
            scaler_g.update()
        else:
            g_loss.backward()
            G_optim.step()
            F_optim.step()

        if i % config.ITERS_PER_LOG == 0:
            logger.info(f"Iter: {i} | C loss: {c_loss.item():.4f} | G loss: {g_loss.item():.4f}")

        if i % config.ITERS_PER_CKPT == 0:
            for net, name in zip([G, F, C_X, C_Y], ["G", "F", "C_X", "C_Y"]):
                path = os.path.join(config.CKPT_DIR, config.TRAIN_STYLE, f"{name}_{i}.pth")
                torch.save(net.state_dict(), path)
            logger.info(f"Saved checkpoints at iteration {i}")

def infer():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    G = Generator(upsample=config.UPSAMPLE).to(device)
    F = Generator(upsample=config.UPSAMPLE).to(device)

    try:
        g_path = os.path.join(config.CKPT_DIR, config.INFER_STYLE, f"G_{config.INFER_ITER}.pth")
        f_path = os.path.join(config.CKPT_DIR, config.INFER_STYLE, f"F_{config.INFER_ITER}.pth")
        G.load_state_dict(torch.load(g_path, map_location=device, weights_only=True))
        F.load_state_dict(torch.load(f_path, map_location=device, weights_only=True))
        logger.info(f"Loaded checkpoints from iteration {config.INFER_ITER}")
    except Exception as e:
        logger.error(f"Failed to load checkpoints for inference: {e}")
        return

    G.eval()
    F.eval()

    # Transformations
    transform_list = []
    if config.IMG_SIZE:
        transform_list.extend([transforms.Resize(config.IMG_SIZE), transforms.CenterCrop(config.IMG_SIZE)])
    transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
    loader = transforms.Compose(transform_list)

    img_path = os.path.join(config.IN_IMG_DIR, config.IMG_NAME)
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = loader(img).unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return

    with torch.no_grad():
        # Using the appropriate generator for stylization
        # In CycleGAN, usually F is Y -> X and G is X -> Y
        # For inference, users typically want the stylization (e.g. Photo -> Van Gogh)
        # Which is G (X -> Y) in this specific implementation's comments
        sty_img = G(img_tensor)
        rec_img = F(sty_img)

    # Naming and paths
    base_name = os.path.splitext(config.IMG_NAME)[0]
    ext = os.path.splitext(config.IMG_NAME)[1]
    iter_k = config.INFER_ITER // 1000
    
    sty_name = f"sty_{base_name}_{config.INFER_STYLE}_{iter_k}k{ext}"
    rec_name = f"rec_{base_name}_{config.INFER_STYLE}_{iter_k}k{ext}"
    
    sty_path = os.path.join(config.SAMPLE_DIR, config.INFER_STYLE, config.OUT_STY_DIR, sty_name)
    rec_path = os.path.join(config.SAMPLE_DIR, config.INFER_STYLE, config.OUT_REC_DIR, rec_name)
    
    vutils.save_image(sty_img, sty_path, normalize=True)
    vutils.save_image(rec_img, rec_path, normalize=True)
    logger.info(f"Saved: {sty_path}\nSaved: {rec_path}")

if __name__ == "__main__":
    setup_directories()
    if config.TRAIN:
        train()
    else:
        infer()
