from utils.networkHelper import *


def cosine_lamda_schedule(timesteps, s=0.008, **kwargs):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    lamdas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(lamdas, 0.0001, 0.9999)


def linear_lamda_schedule(timesteps, lamda_start=0.0001, lamda_end=0.9999):
    return torch.linspace(lamda_start, lamda_end, timesteps)


def quadratic_lamda_schedule(timesteps, lamda_start=0.0001, lamda_end=0.9999):
    return torch.linspace(lamda_start**0.5, lamda_end**0.5, timesteps) ** 2


def sigmoid_lamda_schedule(timesteps, lamda_start=0.0001, lamda_end=0.9999):
    lamdas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(lamdas) * (lamda_end - lamda_start) + lamda_start


class WeightSchedule(nn.Module):
    def __init__(self, schedule_name="linear_lamda_schedule", lamda_start=None, lamda_end=None):
        super(WeightSchedule, self).__init__()

        self.schedule_name = schedule_name

        # 定义函数字典
        lamda_schedule_dict = {'linear_lamda_schedule': linear_lamda_schedule,
                              'cosine_lamda_schedule': cosine_lamda_schedule,
                              'quadratic_lamda_schedule': quadratic_lamda_schedule,
                              'sigmoid_lamda_schedule': sigmoid_lamda_schedule}

        # 根据输入字符串选择相应的函数
        if schedule_name in lamda_schedule_dict:
            self.selected_schedule = lamda_schedule_dict[schedule_name]
        else:
            raise ValueError('Function not found in dictionary')

        if lamda_end and lamda_start is None and schedule_name != "cosine_beta_schedule":
            self.lamda_start = 0.0001
            self.lamda_end = 0.9999
        else:
            self.lamda_start = lamda_start
            self.lamda_end = lamda_end

    def forward(self, timesteps):
        # print(self.beta_start)
        return self.selected_schedule(timesteps=timesteps) if self.schedule_name == "cosine_lamda_schedule" \
            else self.selected_schedule(timesteps=timesteps, lamda_start=self.lamda_start, lamda_end=self.lamda_end)



