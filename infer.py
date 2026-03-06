# -*- coding: utf-8 -*-

import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image
import os
import logging
from accelerate import Accelerator

from networks import Generator
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories(accelerator):
    """Ensure all necessary directories exist."""
    if accelerator.is_main_process:
        dirs_to_make = [config.SAMPLE_DIR]
        for style in config.STYLE_NAMES:
            style_dir = os.path.join(config.SAMPLE_DIR, style)
            dirs_to_make.extend([
                style_dir,
                os.path.join(style_dir, config.OUT_STY_DIR),
                os.path.join(style_dir, config.OUT_REC_DIR)
            ])
        
        for d in dirs_to_make:
            os.makedirs(d, exist_ok=True)

def infer():
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        logger.info(f"Using device: {device}")
    
    gen_a2b = Generator(upsample=config.UPSAMPLE).to(device)
    gen_b2a = Generator(upsample=config.UPSAMPLE).to(device)

    try:
        a2b_path = os.path.join(config.CKPT_DIR, config.INFER_STYLE, f"gen_a2b_{config.INFER_ITER}.pth")
        b2a_path = os.path.join(config.CKPT_DIR, config.INFER_STYLE, f"gen_b2a_{config.INFER_ITER}.pth")
        
        # Load state dicts
        gen_a2b.load_state_dict(torch.load(a2b_path, map_location=device, weights_only=True))
        gen_b2a.load_state_dict(torch.load(b2a_path, map_location=device, weights_only=True))
        
        if accelerator.is_main_process:
            logger.info(f"Loaded checkpoints from iteration {config.INFER_ITER}")
    except Exception as e:
        if accelerator.is_main_process:
            logger.error(f"Failed to load checkpoints for inference: {e}")
        return

    gen_a2b.eval()
    gen_b2a.eval()

    # Prepare for inference (mostly for device placement if multi-gpu, though infer is usually single-gpu)
    gen_a2b, gen_b2a = accelerator.prepare(gen_a2b, gen_b2a)

    # Transformations
    transform_list = []
    if config.IMG_SIZE:
        transform_list.extend([transforms.Resize(config.IMG_SIZE), transforms.CenterCrop(config.IMG_SIZE)])
    transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
    loader = transforms.Compose(transform_list)

    img_path = os.path.join(config.IN_IMG_DIR, config.IMG_NAME)
    if accelerator.is_main_process:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = loader(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                sty_img = gen_a2b(img_tensor)
                rec_img = gen_b2a(sty_img)

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
        except Exception as e:
            logger.error(f"Inference error: {e}")

if __name__ == "__main__":
    from accelerate import Accelerator
    acc = Accelerator()
    setup_directories(acc)
    infer()
