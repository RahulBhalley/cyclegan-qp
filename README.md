# Artist Style Transfer Via Quadratic Potential (2025 Modernized)

[**Rahul Bhalley**](https://github.com/rahulbhalley) and [Jianlin Su](https://github.com/bojone)

[arXiv paper](https://arxiv.org/abs/1902.11108)

### Abstract

In this paper we address the problem of artist style transfer where the painting style of a given artist is applied on a real world photograph. We train our neural networks in adversarial setting via recently introduced quadratic potential divergence for a stable learning process. To further improve the quality of generated artist stylized images we also integrate some of the recently introduced deep learning techniques in our method. To our best knowledge this is the first attempt towards artist style transfer via quadratic potential divergence. We provide some stylized image samples in the supplementary material. The source code for experimentation was written in [PyTorch](https://pytorch.org) and is available online in my [GitHub repository](https://github.com/rahulbhalley/cyclegan-qp).

If you find our work or this repository helpful, please consider citing:
```bibtex
@article{bhalley2019artist,
  title={Artist Style Transfer Via Quadratic Potential},
  author={Bhalley, Rahul and Su, Jianlin},
  journal={arXiv preprint arXiv:1902.11108},
  year={2019}
}
```

### 🚀 2025 Modernization Features

The codebase has been upgraded with the following state-of-the-art GAN training techniques:

*   **🤗 Accelerate Integration:** Seamlessly scale training across CPU, GPU (single/multi), and TPU with built-in support for mixed precision (FP16/BF16).
*   **Differentiable Augmentation (DiffAugment):** Dramatically improves data efficiency and prevents discriminator overfitting on small artist datasets.
*   **Self-Attention (SAGAN):** Integrated into both Generator and Critic to capture long-range spatial dependencies for more coherent global structures.
*   **PyTorch 2.5+ Optimizations:**
    *   **AMP (via Accelerate):** Automatic Mixed Precision for faster training.
    *   **`torch.compile`:** JIT compilation for optimized execution kernels.
    *   **Channels Last:** Optimized memory format for NVIDIA Tensor Cores.
*   **Architectural Upgrades:**
    *   **GELU Activations:** Smoother gradient flow than standard ReLU.
    *   **Residual Scaling:** Improved stability for deep transformer blocks.
    *   **Instance Normalization:** Standardized for high-quality style transfer.

### Prerequisites

- Python (version >= 3.10)
- [PyTorch](https://github.com/pytorch/pytorch) (version >= 2.4.0)
- [Torchvision](https://github.com/pytorch/vision) (version >= 0.19.0)
- [Accelerate](https://github.com/huggingface/accelerate) (version >= 0.30.0)

Install dependencies via:
```bash
pip install -r requirements.txt
```

### Usage

1. **Clone & Setup:**
   ```bash
   git clone https://github.com/rahulbhalley/cyclegan-qp.git
   cd cyclegan-qp
   ```

2. **Download Datasets:**
   ```bash
   bash download_dataset.sh ukiyoe2photo
   ```

3. **Configure Accelerate (First time only):**
   ```bash
   accelerate config
   ```

4. **Run:**
   
   To train the network:
   ```bash
   accelerate launch train.py
   ```

   To perform inference (stylization):
   ```bash
   python infer.py
   ```

### Configurations (`config.py`)

The project uses a structured `Config` dataclass for all hyperparameters.

| Category | Variable | Description |
| :--- | :--- | :--- |
| **Optimization** | `MIXED_PRECISION` | Mixed precision mode ("no", "fp16", "bf16") |
| | `USE_COMPILE` | Enable `torch.compile` for speed |
| | `MATMUL_PRECISION` | Set to 'high' or 'medium' for Tensor Core boost |
| **GAN Tricks** | `USE_INSTANCE_NORM` | Use InstanceNorm instead of BatchNorm |
| | `UPSAMPLE` | Use NN-Upsample + Conv to avoid checkerboard artifacts |
| **Data** | `BATCH_SIZE` | Standard batch size (default: 4) |
| | `LOAD_DIM / CROP_DIM` | Image resizing and cropping dimensions |
| **Losses** | `LAMBDA` | Quadratic Potential penalty weight |
| | `CYC_WEIGHT` | Cycle-consistency weight (default: 10.0) |
| | `ID_WEIGHT` | Identity loss weight (default: 0.5) |

### 🛠️ Code Acknowledgments

This refactored implementation incorporates high-quality modules from the open-source community:

*   **DiffAugment Implementation:** The differentiable augmentation logic in `diff_augment.py` is based on the official implementation by **MIT HAN Lab**: [mit-han-lab/data-efficient-gans](https://github.com/mit-han-lab/data-efficient-gans).
*   **Self-Attention Implementation:** The `SelfAttention` module in `networks.py` follows the architectural standards established in the PyTorch port by **heykeetae**: [heykeetae/Self-Attention-GAN](https://github.com/heykeetae/Self-Attention-GAN).
*   **CycleGAN-QP Core:** The foundational architecture and Quadratic Potential implementation are based on the original work by **Rahul Bhalley**: [rahulbhalley/cyclegan-qp](https://github.com/rahulbhalley/cyclegan-qp).

### 📚 References

This implementation integrates techniques from the following foundational papers:

1.  **CycleGAN-QP (Ours):** Bhalley, R., & Su, J. (2019). [Artist Style Transfer Via Quadratic Potential](https://arxiv.org/abs/1902.11108). *arXiv*.
2.  **DiffAugment:** Zhao, S., et al. (2020). [Differentiable Augmentation for Data-Efficient GAN Training](https://arxiv.org/abs/2006.10738). *NeurIPS*.
3.  **Self-Attention GAN (SAGAN):** Zhang, H., et al. (2019). [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318). *ICML*.
4.  **GELU:** Hendrycks, D., & Gimpel, K. (2016). [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415). *arXiv*.
5.  **Deconvolution Checkerboard:** Odena, A., et al. (2016). [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/). *Distill*.

---

### Results

#### Real Image to Stylized Image
![](https://github.com/rahulbhalley/cyclegan-qp/raw/main/assets/grid_sty.jpg)

#### Stylized Image to Real Image
![](https://github.com/rahulbhalley/cyclegan-qp/raw/main/assets/grid_rec.jpg)
