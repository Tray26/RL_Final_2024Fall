import torch
import torch.nn as nn
import numpy as np

from gen_ddpm import MLPDiffusion, sigmoid_beta_schedule

class DDIM(nn.Module):
    def __init__(self, train_timestep):
        super().__init__()
        self.denoise_net = MLPDiffusion()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scheduler(train_timestep)
        
    def scheduler(self, T):
        # assert beta0 < betaT < 1.0, "beta1 and beta2 must be in (0, 1)"
        beta_schedule = sigmoid_beta_schedule(T)
        alpha_schedule = 1-beta_schedule
        log_alpha = torch.log(alpha_schedule)
        alpha_bar = torch.cumsum(log_alpha, dim=0).exp()
        
        self.betas = beta_schedule.to(self.device)
        self.alpha_bars = alpha_bar.to(self.device)
        self.alphas = alpha_schedule.to(self.device)
        
    def ddim_sample(self, x_T, device, noise_step, eta=0, ddim_step=50):
        sample_ts = (
            np.linspace(0, noise_step-1, ddim_step).round()[::-1]
            .copy()
            .astype(np.int64)
        )
        sample_ts = torch.from_numpy(sample_ts).to(device)
        x_t = x_T.to(device)
        sample_size = x_T.shape[0]
        step_imgs = []
        for i in range(1, ddim_step):
            cur_t = sample_ts[i-1]
            prev_t = sample_ts[i]
            print(f"sampling timestep {cur_t}", end="\r")
            
            ab_cur = self.alpha_bars[cur_t]
            ab_prev = self.alpha_bars[prev_t] if prev_t >= 0 else torch.tensor(1.0)
            t_tensor = torch.tensor([cur_t] * sample_size,
                                dtype=torch.long).to(device).unsqueeze(1)
            eps = self.denoise_net(x_t, t_tensor)

            beta_tilde = eta**2 * ((1 - ab_prev) / (1 - ab_cur)) * (1 - (ab_cur/ab_prev) ) # beta^tilde
            first_term = (ab_prev / ab_cur)**0.5 * x_t
            second_term = ((1 - ab_prev - beta_tilde)**0.5 -
                            (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
            third_term = beta_tilde**0.5 * torch.randn_like(x_t)
            x_t = first_term + second_term + third_term
            step_imgs.append(x_t.detach().cpu().numpy())
        