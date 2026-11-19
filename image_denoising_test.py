import os
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
from torch.optim import Adam
from utils.networkHelper import *
from torchvision.transforms import ToPILImage
from noisePredictModels.Unet.UNet import Unet
from utils.trainNetworkHelper import DenoisingDiffusionTrainer
from diffusionModels.denoisingDiffusion.denoisingDiffusion import DenoinsingDiffusionModel


imagenet_data = torchvision.datasets.FashionMNIST("./dataset/fashion/raw/", train=True, download=True, transform=transforms.ToTensor())
stacked_ds = StackedNoisyDataset(imagenet_data, snr_db_range=(-5.0, 5.0), clamp=True, sample_in_db=True)

image_size = 28
input_channels = 2
output_channels = 1
batch_size = 64

data_loader = torch.utils.data.DataLoader(stacked_ds,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)
batch, labels = next(iter(data_loader))
# print("batch shape:", batch.shape)  # 期望 (8, 2, 28, 28) 对于 MNIST (C=1)

# # 保存第一个样本的可视化（左右并排：clean | noisy）
# sample = batch[2]  # (2, H, W)
# clean = sample[0:1]   # (1,H,W)
# noisy = sample[1:2]   # (1,H,W)
# side_by_side = torch.cat([clean, noisy], dim=2)  # (1, H, 2*W)
# torchvision.utils.save_image(side_by_side, "mnist_clean_vs_noisy.png")
# print("Saved mnist_clean_vs_noisy.png")

device = "cuda" if torch.cuda.is_available() else "cpu"
dim_mults = (1, 2, 4,)

denoise_model = Unet(
    dim=image_size,
    input_channels=input_channels,
    output_channels=output_channels,
    dim_mults=dim_mults
)

timesteps = 200
DDPM = DenoinsingDiffusionModel(var_schedule_name="linear_beta_schedule",
                      timesteps=timesteps,
                      beta_start=0.0001,
                      beta_end=0.0095,
                      denoise_model=denoise_model).to(device)

optimizer = Adam(DDPM.parameters(), lr=1e-3)
epoches = 20

Trainer = DenoisingDiffusionTrainer(epoches=epoches,
                                 train_loader=data_loader,
                                 optimizer=optimizer,
                                 device=device,
                                 timesteps=timesteps,
                                 use_ema=True,
                                 ema_decay=0.995)

# 训练参数设置
root_path = "./saved_denoising_train_models"
setting = "imageSize{}_channels{}_dimMults{}_timeSteps{}_scheduleName{}_weightScheduleName{}".format(image_size, output_channels, dim_mults, timesteps, "linear_beta_schedule", "linear_lamda_schedule")

saved_path = os.path.join(root_path, setting)
if not os.path.exists(saved_path):
    os.makedirs(saved_path)


# 训练好的模型加载，如果模型是已经训练好的，则可以将下面两行代码取消注释
# best_model_path = saved_path + '/' + 'BestModel.pth'
# DDPM.load_state_dict(torch.load(best_model_path))

# 如果模型已经训练好则注释下面这行代码，反之则注释上面两行代码
DDPM = Trainer(DDPM, model_save_path=saved_path)

batch, labels = next(iter(data_loader))
print("batch shape:", batch.shape)
image_clean = batch[:,0,:,:]
image_noisy = batch[:,1,:,:]
# 采样:sample 64 images
samples = DDPM(mode="generate", x_noised=image_noisy.to(device))
print("samples shape:", samples[-1].shape)
# 随机挑一张显示
denoised_all = samples[-1]
n = 6  # 要展示的图片数量，按需修改
bs = image_clean.shape[0]
n = min(n, bs, denoised_all.shape[0])
idxs = np.random.choice(bs, size=n, replace=False).tolist()
for i, idx in enumerate(idxs):
    denoised_image = denoised_all[idx].reshape(output_channels, image_size, image_size)
    figtest = reverse_transform_test(torch.from_numpy(denoised_image))
    figclean = reverse_transform_test(image_clean[idx].unsqueeze(0))
    fignoisy = reverse_transform_test(image_noisy[idx].unsqueeze(0))
    combined = np.concatenate((figclean, fignoisy, figtest), axis=1)
    if i == 0:
        final_image = combined
    else:
        final_image = np.concatenate((final_image, combined), axis=0)
final_pil_image = ToPILImage()(final_image)
final_pil_image.show()
final_pil_image.save("denoising_results_4.png")