from utils.networkHelper import *
from diffusionModels.denoisingDiffusion.varianceSchedule import VarianceSchedule


class DenoinsingDiffusionModel(nn.Module):
    def __init__(self,
                 var_schedule_name="linear_beta_schedule",
                 timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 denoise_model=None):
        super(DenoinsingDiffusionModel, self).__init__()

        self.denoise_model = denoise_model

        # 方差生成
        variance_schedule_func = VarianceSchedule(schedule_name=var_schedule_name, beta_start=beta_start, beta_end=beta_end)
        self.timesteps = timesteps
        self.betas = variance_schedule_func(timesteps)
        # define alphas
        self.alphas = 1. - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_one_minus_alphas_cumprod = 1.0 / torch.sqrt(1. - self.alphas_cumprod)

        # 时间步权重生成
        # weight_schedule_func = WeightSchedule(schedule_name=weight_schedule_name, lamda_start=lamda_start, lamda_end=lamda_end)
        # self.lamdas = weight_schedule_func(timesteps)
        self.lamdas = torch.sqrt((1. - self.alphas_cumprod) / self.sqrt_alphas_cumprod)
        self.one_minus_lamdas = 1. - self.lamdas
        self.deltas = (1. - self.alphas_cumprod) - torch.pow(self.lamdas, 2) * self.alphas_cumprod
        self.sqrt_deltas = torch.sqrt(self.deltas)

    def q_sample(self, x_start, x_noised, t, noise=None):
        # forward diffusion (using the nice property)
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        one_minus_lamda_t = extract(self.one_minus_lamdas, t, x_start.shape)
        lamda_t = extract(self.lamdas, t, x_start.shape)
        sqrt_deltas_t = extract(self.sqrt_deltas, t, x_start.shape)

        return one_minus_lamda_t * sqrt_alphas_cumprod_t * x_start + lamda_t * sqrt_alphas_cumprod_t * x_noised + sqrt_deltas_t * noise

    def compute_loss(self, x_start, x_noised, t, noise=None, loss_type="l1"):
        sqrt_recip_one_minus_alphas_cumprod_t = extract(self.sqrt_recip_one_minus_alphas_cumprod, t, x_start.shape)
        lamda_t = extract(self.lamdas, t, x_start.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_deltas_t = extract(self.sqrt_deltas, t, x_start.shape)
        if noise is None:
            noise = torch.randn_like(x_start)
        noise_condition = sqrt_recip_one_minus_alphas_cumprod_t * (
            lamda_t * sqrt_alphas_cumprod_t * (x_noised - x_start) 
            + sqrt_deltas_t * noise)
        x_noisy = self.q_sample(x_start=x_start, x_noised=x_noised, t=t, noise=noise)
        x_condition=torch.stack((x_noisy,x_noised),dim=1)
        predicted_noise = self.denoise_model(x_condition, t)
        # 去掉 noise_condition 的通道维度（从 [128,1,28,28] → [128,28,28]）
        predicted_noise = predicted_noise.squeeze(1)  # 挤压第1维（通道维）
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, x, x_noised, t, t_index):
        alphas_t = extract(self.alphas, t, x.shape)
        lamda_t = extract(self.lamdas, t, x.shape)
        lamda_t_delayed = extract(self.lamdas, t-1, x.shape)
        delta_t = extract(self.deltas, t, x.shape)
        delta_t_delayed = extract(self.deltas, t-1, x.shape)
        sqrt_alphas_t = extract(self.sqrt_alphas, t, x.shape)
        sqrt_alphas_cumprod_t_delayed = extract(self.sqrt_alphas_cumprod, t-1, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        delta_t_condition = delta_t - torch.pow(((1. - lamda_t) / (1 - lamda_t_delayed)), 2) * alphas_t * delta_t_delayed
        sqrt_delta_t_hat = torch.sqrt(delta_t_condition * delta_t / delta_t_delayed)
        
        Psi_x = (
            (delta_t_delayed * (1. - lamda_t) * sqrt_alphas_t) / (delta_t * (1. - lamda_t_delayed))
            + (1. - lamda_t_delayed) * delta_t_condition / (delta_t * sqrt_alphas_t)
        )

        Psi_x_noise = (
            lamda_t_delayed * delta_t - (lamda_t * (1. - lamda_t) * alphas_t * delta_t_delayed) / (1. - lamda_t_delayed)
        ) * sqrt_alphas_cumprod_t_delayed / delta_t

        Psi_noise = (1. - lamda_t_delayed) * delta_t_condition * sqrt_one_minus_alphas_cumprod_t / (delta_t * sqrt_alphas_t)
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        x_condition=torch.stack((x,x_noised),dim=1)
        model_mean = Psi_x * x + Psi_x_noise * x_noised - Psi_noise * self.denoise_model(x_condition, t).squeeze(1)
        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + sqrt_delta_t_hat * noise

    @torch.no_grad()
    def p_sample_loop(self, x_noised):
        device = next(self.denoise_model.parameters()).device

        b = x_noised.shape[0]
        T = torch.full((b,), self.timesteps - 1, device=device, dtype=torch.long)
        sqrt_alphas_T = extract(self.sqrt_alphas_cumprod, T, x_noised.shape)
        delta_T = extract(self.deltas, T, x_noised.shape)
        # start from pure noise (for each example in the batch)
        img = sqrt_alphas_T * x_noised + torch.sqrt(delta_T) * torch.randn_like(x_noised)
        imgs = []
        imgs.append(img.cpu().numpy())
        for i in tqdm(reversed(range(1, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(img, x_noised, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, x_noised):
        return self.p_sample_loop(x_noised)

    def forward(self, mode, **kwargs):
        if mode == "train":
            # 先判断必须参数
            if "x_start" and "t" and "x_noised" in kwargs.keys():
                # 接下来判断一些非必选参数
                if "loss_type" and "noise" in kwargs.keys():
                    return self.compute_loss(x_start=kwargs["x_start"], x_noised=kwargs["x_noised"], t=kwargs["t"],
                                             noise=kwargs["noise"], loss_type=kwargs["loss_type"])
                elif "loss_type" in kwargs.keys():
                    return self.compute_loss(x_start=kwargs["x_start"], x_noised=kwargs["x_noised"], t=kwargs["t"],
                                             loss_type=kwargs["loss_type"])
                elif "noise" in kwargs.keys():
                    return self.compute_loss(x_start=kwargs["x_start"], x_noised=kwargs["x_noised"], t=kwargs["t"],
                                             noise=kwargs["noise"])
                else:
                    return self.compute_loss(x_start=kwargs["x_start"], x_noised=kwargs["x_noised"], t=kwargs["t"])

            else:
                raise ValueError("扩散去噪模型在训练时必须传入参数x_start、x_noised、t！")

        elif mode == "generate":
            if "x_noised" in kwargs.keys():
                # x_noised shape (batch_size, channels, height, width)
                return self.sample(x_noised=kwargs["x_noised"])
            else:
                raise ValueError("扩散去噪模型在图片去噪时必须传入x_noised参数")
        else:
            raise ValueError("mode参数必须从{train}和{generate}两种模式中选择")





