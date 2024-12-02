import os
import torch

def parse_trajectories(trajectories, state_dim, done):
    parsed_data = {'obs': [], 'actions': [], 'next_obs': []}
    parsed_data['obs'] = trajectories[:, :state_dim].clone().detach()
    parsed_data['next_obs'] = trajectories[1:, :state_dim].clone().detach()
    parsed_data['actions'] = trajectories[:, state_dim:].clone().detach()
    parsed_data['done'] = done[:].clone().detach()
    return parsed_data

def denorm_vec(x, mean, std):
    denorm = x*(std + 1e-8) + mean
    return denorm

def norm_vec(x, mean, std):
    obs_x = torch.clamp(
        (x - mean) / (std + 1e-8),
        -10.0,
        10.0,
    )
    return obs_x

def sigmoid_beta_schedule(timesteps, start=0.0001, end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (end - start) + start
