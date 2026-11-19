"""Evaluate reconstruction quality (PSNR / SSIM) of the conditional diffusion model.

Usage
-----
    python evaluate.py --ckpt <BestModel.pth> --timesteps 200
    python evaluate.py --timesteps 1000 --beta-end 0.02 --snrs -5 0 5 10 --num-images 64 --csv results.csv

The script reports, for each SNR, the PSNR of the degraded input (baseline)
and the PSNR / SSIM of the reconstructed output, so the gain can be read
directly instead of being estimated from a picture.
"""

import argparse
import csv
import os

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from diffusionModels.denoisingDiffusion.denoisingDiffusion import DenoinsingDiffusionModel
from noisePredictModels.Unet.UNet import Unet
from utils.networkHelper import add_gaussian_noise_snr

IMAGE_SIZE = 28
DEFAULT_CKPT = (
    "saved_denoising_train_models/"
    "imageSize28_channels1_dimMults(1, 2, 4)_timeSteps200_"
    "scheduleNamelinear_beta_schedule_weightScheduleNamelinear_lamda_schedule/BestModel.pth"
)


def parse_args():
    p = argparse.ArgumentParser(description="PSNR/SSIM evaluation for the conditional diffusion model")
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="path to BestModel.pth")
    p.add_argument("--timesteps", type=int, default=200, help="diffusion steps, must match the checkpoint")
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=None,
                   help="defaults to 9.5e-3 for 200 steps and 2e-2 for 1000 steps")
    p.add_argument("--dataset", choices=["fashion", "mnist"], default="fashion")
    p.add_argument("--snrs", type=float, nargs="+", default=[-5.0, 0.0, 5.0, 10.0],
                   help="SNR values in dB")
    p.add_argument("--num-images", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default=None, help="optional path to write a CSV table")
    p.add_argument("--device", default=None)
    return p.parse_args()


def psnr(x, y, eps=1e-8):
    """Mean PSNR over a batch, images expected in [0, 1]."""
    mse = (x - y).pow(2).flatten(1).mean(dim=1)
    return (10.0 * torch.log10(1.0 / (mse + eps))).mean().item()


def gaussian_window(size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    return (g.t() @ g).unsqueeze(0).unsqueeze(0)


def ssim(x, y, window):
    """Mean SSIM over a batch of (N, 1, H, W) images in [0, 1]."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    pad = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=pad)
    mu_y = F.conv2d(y, window, padding=pad)
    sigma_x = F.conv2d(x * x, window, padding=pad) - mu_x ** 2
    sigma_y = F.conv2d(y * y, window, padding=pad) - mu_y ** 2
    sigma_xy = F.conv2d(x * y, window, padding=pad) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.flatten(1).mean(dim=1).mean().item()


def load_dataset(name):
    root = "./dataset/fashion/raw/" if name == "fashion" else "./dataset/MNIST/raw/"
    dataset = "FashionMNIST" if name == "fashion" else "MNIST"
    cls = getattr(torchvision.datasets, dataset)
    return cls(root, train=False, download=True, transform=transforms.ToTensor())


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    beta_end = args.beta_end
    if beta_end is None:
        beta_end = 9.5e-3 if args.timesteps == 200 else 2e-2

    torch.manual_seed(args.seed)

    unet = Unet(dim=IMAGE_SIZE, input_channels=2, output_channels=1, dim_mults=(1, 2, 4))
    model = DenoinsingDiffusionModel(
        var_schedule_name="linear_beta_schedule",
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=beta_end,
        denoise_model=unet,
    ).to(device)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"device={device}  timesteps={args.timesteps}  beta_end={beta_end}  ckpt={args.ckpt}")

    test_set = load_dataset(args.dataset)
    clean = torch.stack([test_set[i][0] for i in range(min(args.num_images, len(test_set)))])
    clean = clean.to(device)  # (N, 1, 28, 28)

    window = gaussian_window(device=device)
    rows = []
    for snr in args.snrs:
        psnr_noisy, psnr_recon, ssim_total = [], [], []
        for start in range(0, clean.shape[0], args.batch_size):
            ref = clean[start:start + args.batch_size]
            degraded = add_gaussian_noise_snr(ref, snr, clamp=True)
            samples = model(mode="generate", x_noised=degraded[:, 0])
            recon = torch.as_tensor(samples[-1], device=device, dtype=torch.float32).clamp(0.0, 1.0)
            recon = recon.unsqueeze(1)

            psnr_noisy.append(psnr(ref, degraded))
            psnr_recon.append(psnr(ref, recon))
            ssim_total.append(ssim(ref, recon, window))

        n_batches = len(psnr_noisy)
        avg_noisy = sum(psnr_noisy) / n_batches
        avg_recon = sum(psnr_recon) / n_batches
        avg_ssim = sum(ssim_total) / n_batches
        rows.append((snr, avg_noisy, avg_recon, avg_recon - avg_noisy, avg_ssim))
        print(f"SNR {snr:+5.1f} dB | degraded {avg_noisy:6.2f} dB | "
              f"reconstructed {avg_recon:6.2f} dB | gain {avg_recon - avg_noisy:+6.2f} dB | "
              f"SSIM {avg_ssim:.4f}")

    print("\n| SNR (dB) | degraded PSNR | reconstructed PSNR | gain | SSIM |")
    print("| --- | --- | --- | --- | --- |")
    for snr, a, b, g, s in rows:
        print(f"| {snr:+.1f} | {a:.2f} | {b:.2f} | {g:+.2f} | {s:.4f} |")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["snr_db", "psnr_degraded", "psnr_reconstructed", "gain_db", "ssim"])
            for row in rows:
                writer.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in row])
        print(f"\nsaved: {args.csv}")


if __name__ == "__main__":
    main()
