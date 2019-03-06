# -*- coding: utf-8 -*-

__author__ = "Rahul Bhalley"

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils as vutils

from networks import Generator, Critic

from config import *
from data import *

import os


####################
# Make directories #
####################

try:
    if TRAIN:
        # Checkpoint directories
        if not os.path.exists(CKPT_DIR):
            os.mkdir(CKPT_DIR)
        for style in STYLES:
            if not os.path.exists(os.path.join(CKPT_DIR, style)):
                os.mkdir(os.path.join(CKPT_DIR, style))
    else:
        # Sample directories
        if not os.path.exists(SAMPLE_DIR):
            os.mkdir(SAMPLE_DIR)
        
        for style in STYLES:
            if not os.path.exists(os.path.join(SAMPLE_DIR, style)):
                os.mkdir(os.path.join(SAMPLE_DIR, style))
                # Make three directories
                os.mkdir(os.path.join(SAMPLE_DIR, style, OUT_STY_DIR))  # Stylized images here
                os.mkdir(os.path.join(SAMPLE_DIR, style, OUT_REC_DIR))  # Reconstructed images here
except:
    print("Directories already exist!")


####################
# Load the dataset #
####################

if TRAIN:
    # Make experiments reproducible
    _ = torch.manual_seed(RANDOM_SEED)
    
    # Load the datasets
    X_set, Y_set = load_data()

    # Load infinite data
    X_data = get_infinite_X_data(X_set)
    Y_data = get_infinite_Y_data(Y_set)


########################################################
# Define device, neural nets, losses, optimizers, etc. #
########################################################

# Automatic GPU/CPU device placement
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Networks
C_X = Critic().to(device)   # Criticizes X data
C_Y = Critic().to(device)   # Criticizes Y data
G = Generator(upsample=UPSAMPLE).to(device) # Translates X -> Y
F = Generator(upsample=UPSAMPLE).to(device) # Translates Y -> X

# Losses
l1_loss = nn.L1Loss()

# Optimizers
C_X_optim = optim.Adam(C_X.parameters(), lr=LR, betas=(BETA1, BETA2))
C_Y_optim = optim.Adam(C_Y.parameters(), lr=LR, betas=(BETA1, BETA2))
G_optim =   optim.Adam(G.parameters(),   lr=LR, betas=(BETA1, BETA2))
F_optim =   optim.Adam(F.parameters(),   lr=LR, betas=(BETA1, BETA2))


###############
# Training 🧠 #
###############

def train():

    # Status
    print("Begin training!")

    # Load the checkpoints from `BEGIN_ITER`
    try:
        # Get checkpoint paths
        g_model_path =   os.path.join(CKPT_DIR, TRAIN_STYLE, f"G_{BEGIN_ITER}.pth")
        f_model_path =   os.path.join(CKPT_DIR, TRAIN_STYLE, f"F_{BEGIN_ITER}.pth")
        c_x_model_path = os.path.join(CKPT_DIR, TRAIN_STYLE, f"C_X_{BEGIN_ITER}.pth")
        c_y_model_path = os.path.join(CKPT_DIR, TRAIN_STYLE, f"C_Y_{BEGIN_ITER}.pth")

        # Load parameters from checkpoint paths
        G.load_state_dict(torch.load(g_model_path,     map_location=device))
        F.load_state_dict(torch.load(f_model_path,     map_location=device))
        C_X.load_state_dict(torch.load(c_x_model_path, map_location=device))
        C_Y.load_state_dict(torch.load(c_y_model_path, map_location=device))
        
        # Status
        print(f"Training: Loaded the checkpoints from {BEGIN_ITER}th iteration.")
    except:
        # Status
        print(f"Training: Couldn't load the checkpoints from {BEGIN_ITER}th iteration.")

    # Now finally begin training!
    for i in range(BEGIN_ITER, END_ITER + 1):
        
        # Sample safely
        x, y = safe_sampling(X_data, Y_data, device)

        #################
        # Train Critics #
        #################

        # Update gradient computation:
        # ∙ 👎 Generators
        # ∙ 👍 Critics
        for param in G.parameters():
            param.requires_grad_(False)
        for param in F.parameters():
            param.requires_grad_(False)
        for param in C_X.parameters():
            param.requires_grad_(True)
        for param in C_Y.parameters():
            param.requires_grad_(True)

        for j in range(2):

            # Forward passes:
            # ∙ X -> Y
            # ∙ Y -> X

            # Domain translation: X -> Y
            with torch.no_grad():
                G_x = G(x)      # G(x),     X -> Y
            C_Y_G_x = C_Y(G_x)  # Cy(G(x)), fake score
            C_y_y = C_Y(y)      # Cy(y),    real score

            # Domain translation: Y -> X
            with torch.no_grad():
                F_y = F(y)      # F(y),     Y -> X
            C_X_F_y = C_X(F_y)  # Cx(F(y)), fake score
            C_X_x = C_X(x)      # Cx(x),    real score

            # Zerofy the gradients
            C_X_optim.zero_grad()
            C_Y_optim.zero_grad()

            # Compute the losses:
            # ∙ QP-div loss (critizing x data),     Y -> X
            # ∙ QP-div loss (critizing y data),     X -> Y

            # QP-div loss (critizing x data)
            x_loss = C_X_x - C_X_F_y    # real score - fake score
            if NORM == "l1":
                x_norm = LAMBDA * (x - F_y).abs().mean()
            elif NORM == "l2":
                x_norm = LAMBDA * ((x - F_y)**2).mean().sqrt()
            x_loss = -x_loss + 0.5 * x_loss**2 / x_norm
            x_loss = x_loss.mean()

            # QP-div loss (critizing y data)
            y_loss = C_y_y - C_Y_G_x    # real score - fake score
            if NORM == "l1":
                y_norm = LAMBDA * (y - G_x).abs().mean()
            elif NORM == "l2":
                y_norm = LAMBDA * ((y - G_x)**2).mean().sqrt()
            y_loss = -y_loss + 0.5 * y_loss**2 / y_norm
            y_loss = y_loss.mean()

            # Total loss
            c_loss = x_loss + y_loss

            # Compute gradients
            c_loss.backward()

            # Update the networks
            C_Y_optim.step()
            C_X_optim.step()

        ####################
        # Train Generators #
        ####################

        # Update gradient computation:
        # ∙ 👍 Generators
        # ∙ 👎 Critics
        for param in G.parameters():
            param.requires_grad_(True)
        for param in F.parameters():
            param.requires_grad_(True)
        for param in C_X.parameters():
            param.requires_grad_(False)
        for param in C_Y.parameters():
            param.requires_grad_(False)

        for j in range(1):

            # Forward passes:
            # ∙ X -> Y
            # ∙ Y -> X
            # ∙ X -> Y -> X
            # ∙ Y -> X -> Y

            # Domain translation: X -> Y
            G_x = G(x)          # G(x),     X -> Y
            C_Y_G_x = C_Y(G_x)  # Cy(G(x)), fake score
            C_y_y = C_Y(y)      # Cy(y),    real score

            # Domain translation: Y -> X
            F_y = F(y)          # F(y),     Y -> X
            C_X_F_y = C_X(F_y)  # Cx(F(y)), fake score
            C_X_x = C_X(x)      # Cx(x),    real score

            # Cycle-consistent translations
            F_G_x = F(G_x)      # F(G(x)), X -> Y -> X
            G_F_y = G(F_y)      # G(F(y)), Y -> X -> Y

            # Zerofy the gradients
            G_optim.zero_grad()
            F_optim.zero_grad()

            # Compute the losses:
            # ∙ QP-div loss (critizing x data),     Y -> X
            # ∙ QP-div loss (critizing y data),     X -> Y
            # ∙ Cycle-consistency loss,             || F(G(x)) - x || L1
            # ∙ Cycle-consistency loss,             || G(F(y)) - y || L1
            # ∙ Identity loss,                      || G(x) - y || L1
            # ∙ Identity loss,                      || F(y) - x || L1
            
            # QP-div losses
            x_loss = C_X_x - C_X_F_y        # real score - fake score
            y_loss = C_y_y - C_Y_G_x        # real score - fake score
            x_loss = x_loss.mean()
            y_loss = y_loss.mean()

            # Cycle-consistency losses
            x_cyc_loss = l1_loss(F_G_x, x)  # || F(G(x)) - x || L1
            y_cyc_loss = l1_loss(G_F_y, y)  # || G(F(y)) - y || L1
            x_cyc_loss = x_cyc_loss.mean()
            y_cyc_loss = y_cyc_loss.mean()
            
            # Identity losses
            x_id_loss = l1_loss(G_x, y)     # || G(x) - y || L1
            y_id_loss = l1_loss(F_y, x)     # || F(y) - x || L1
            x_id_loss = x_id_loss.mean()
            y_id_loss = y_id_loss.mean()

            # Total loss
            g_loss = x_loss + y_loss
            g_loss += CYC_WEIGHT * (x_cyc_loss + y_cyc_loss)
            g_loss += ID_WEIGHT * (x_id_loss + y_id_loss)

            # Compute gradients
            g_loss.backward()

            # Update the networks
            G_optim.step()
            F_optim.step()

        #############
        # Log stats #
        #############

        if i % ITERS_PER_LOG == 0:
            # Status
            print(f"iter: {i} c_loss: {c_loss} g_loss: {g_loss}")

        if i % ITERS_PER_CKPT == 0:
            # Get checkpoint paths
            g_model_path =   os.path.join(CKPT_DIR, TRAIN_STYLE, f"G_{i}.pth")
            f_model_path =   os.path.join(CKPT_DIR, TRAIN_STYLE, f"F_{i}.pth")
            c_x_model_path = os.path.join(CKPT_DIR, TRAIN_STYLE, f"C_X_{i}.pth")
            c_y_model_path = os.path.join(CKPT_DIR, TRAIN_STYLE, f"C_Y_{i}.pth")

            # Save the checkpoints
            torch.save(G.state_dict(),   g_model_path)
            torch.save(F.state_dict(),   f_model_path)
            torch.save(C_X.state_dict(), c_x_model_path)
            torch.save(C_Y.state_dict(), c_y_model_path)

            # Status
            print(f"Saved checkpoints at {i}th iteration.")
    # Status
    print("Finished Training!")


################
# Inference 🧠 #
################

def infer(iteration, style, img_name, in_img_dir, out_rec_dir, out_sty_dir, img_size=None):
    
    # Set neural nets to evaluation mode
    G.eval()
    F.eval()

    # Try loading models from checkpoints at `iteration`
    try:
        # Get checkpoint paths
        g_model_path = os.path.join(CKPT_DIR, style, f"G_{iteration}.pth")
        f_model_path = os.path.join(CKPT_DIR, style, f"F_{iteration}.pth")
        
        # Load parameters from checkpoint paths
        G.load_state_dict(torch.load(g_model_path, map_location=device))
        F.load_state_dict(torch.load(f_model_path, map_location=device))
        
        # Status
        print(f"Inference: Loaded the checkpoints from {iteration}th iteration.")
    except:
        # Status
        print(f"Inference: Couldn't load the checkpoints from {iteration}th iteration.")
        raise

    # Minor transforms
    if img_size == None:
        loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
    else:
        loader = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])

    from PIL import Image

    def image_loader(image_name):
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)  # Add a fake batch dimension
        return image.to(device, torch.float)
    
    # style_a = image_loader(out_img_path)
    in_img_path = os.path.join(in_img_dir, img_name)
    in_img = image_loader(in_img_path)
    
    with torch.no_grad():
        print("Stylization")
        sty_img = F(in_img)     # Y -> X
        print("Reconstruction")
        rec_img = G(sty_img)    # X -> Y
    
    # WARNING: Please do not change this code snippet with a closed mind. 🤪👻
    iteration = int(iteration / 1000)
    only_img_name = img_name.split('.')[0]
    img_type = img_name.split('.')[1]

    # Set up names
    out_sty_name = f"sty_{only_img_name}_{style}_{iteration}k.{img_type}"
    out_rec_name = f"rec_{only_img_name}_{style}_{iteration}k.{img_type}"
    
    # Set up paths
    sty_path = os.path.join(SAMPLE_DIR, style, out_sty_dir, out_sty_name)
    rec_path = os.path.join(SAMPLE_DIR, style, out_rec_dir, out_rec_name)
    
    # Save image grids
    vutils.save_image(sty_img, sty_path, normalize=True)
    vutils.save_image(rec_img, rec_path, normalize=True)
    
    # Status
    print(f"Saved {rec_path}")
    print(f"Saved {sty_path}")


if __name__ == "__main__":
    
    if TRAIN:
        train()
    else:
        infer(
            iteration=INFER_ITER,
            style=INFER_STYLE,
            img_name=IMG_NAME,
            in_img_dir=IN_IMG_DIR,
            out_rec_dir=OUT_REC_DIR,
            out_sty_dir=OUT_STY_DIR,
            img_size=IMG_SIZE
        )